from typing import Dict, Any, Optional, Iterable, Tuple, List

import numpy as np
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.ent_model import EntailmentModel
from dataset.entailment_dataset import EntailmentDataset, Sample

EVAL_BSIZE = 32


def make_dataloader(dataset: EntailmentDataset, batch_size: int, split: str) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), collate_fn=lambda x: x)


class ModelInstructor:
    def __init__(self, model: EntailmentModel, device, cfg_optimizer: Dict[str, Any]):
        self.model = model
        self.device = device
        self.cfg_optimizer = cfg_optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_optimizer['learning_rate'])
        self.criterion = torch.nn.BCELoss()
        self.tracked_metrics = None
        self._refresh_tracked_metrics()

    def _refresh_tracked_metrics(self):
        self.tracked_metrics = {metric: [] for metric in ['loss', 'acc']}

    def calc_metrics(self, batch_predictions, batch_labels) -> Tuple[torch.Tensor, Dict[str, float]]:
        bsize = batch_predictions.shape[0]
        loss = self.criterion(batch_predictions, batch_labels)

        predicted_inds = torch.nn.functional.one_hot(torch.argmax(batch_predictions, dim=1), num_classes=2)
        correct = torch.sum(predicted_inds * batch_labels).item()
        acc = correct / bsize

        metrics = {
                'loss': loss.item(),
                'acc': acc
        }
        return loss, metrics

    def _training_display_status(self):
        loss = np.mean(self.tracked_metrics['loss'])
        acc = np.mean(self.tracked_metrics['acc'])
        return f'loss: {loss:.3f}\tacc: {acc*100:.2f}%'

    def process_batch(self, batch: Iterable[Sample]) -> Tuple[List[str], List[str], torch.tensor]:
        batch_premises = [sample.premise for sample in batch]
        batch_premises = [premise_triple[1] for premise_triple in batch_premises]

        batch_hypotheses = [sample.hypothesis for sample in batch]
        batch_hypotheses = [' '.join(hypothesis_triple) for hypothesis_triple in batch_hypotheses]

        truth_values = [int(sample.truth_value) for sample in batch]
        batch_labels = torch.nn.functional.one_hot(torch.tensor(truth_values), num_classes=2).float().to(self.device)
        return batch_premises, batch_hypotheses, batch_labels

    def step(self, batch: Iterable[Sample], training=True):
        # extract data
        batch_premises, batch_hypotheses, batch_labels = self.process_batch(batch)

        # forward pass
        self.model.train()
        batch_predictions = self.model(batch_premises, batch_hypotheses)

        # eval results
        loss, batch_metrics = self.calc_metrics(batch_predictions, batch_labels)

        if training:
            # record metrics over an epoch
            for metric in batch_metrics:
                self.tracked_metrics[metric].append(batch_metrics[metric])

            # backprop, update, and zero-out
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.zero_grad()
            self.optimizer.zero_grad()

        return batch_metrics

    def train_model(self, datasets: Dict[str, Optional[EntailmentDataset]]):

        batch_size = self.cfg_optimizer['batch_size']
        train_data = make_dataloader(datasets['train'], batch_size, 'train')

        max_train_steps = self.cfg_optimizer['max_train_steps']
        epoch = 0
        with tqdm(total=max_train_steps, ncols=120) as pbar:
            train_step = 0
            while train_step < max_train_steps:
                epoch += 1
                pbar.set_description(f'Epoch {epoch} > training step')
                self._refresh_tracked_metrics()
                for batch_num, batch in enumerate(train_data):
                    # predict and backprop
                    self.step(batch)

                    # update train progress
                    train_step += 1
                    pbar.update(1)
                    pbar.set_postfix_str(self._training_display_status())

                    # quit if done
                    if train_step == max_train_steps:
                        break

                if datasets['dev']:
                    print()
                    self.eval_model(datasets['dev'], 'dev')

    def eval_model(self, dataset: EntailmentDataset, split: str):
        print(f'Evaluating {split}...')
        self.model.eval()
        eval_data = make_dataloader(dataset, EVAL_BSIZE, split)

        all_metrics = {}
        num_samples = 0
        for batch_num, batch in enumerate(eval_data):
            this_batch_size = len(batch)
            num_samples += this_batch_size
            batch_metrics = self.step(batch, training=False)
            for metric in batch_metrics:
                new_metric_val = (batch_metrics[metric] * this_batch_size)
                all_metrics[metric] = all_metrics[metric] + new_metric_val if metric in all_metrics else new_metric_val

        for metric in all_metrics:
            all_metrics[metric] /= num_samples

        loss = all_metrics['loss']
        acc = all_metrics['acc']

        print('-' * 100)
        print(f'{split} |\tloss: {loss:.3f}\tacc: {acc*100:.1f}')
        print('-' * 100)

