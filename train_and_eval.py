import os
import operator
from typing import Dict, Any, Optional, Iterable, Tuple, List

import numpy as np
from sklearn import metrics
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from utils import print_bar, colors
from model.ent_model import ProjectorModel
from dataset.entailment_dataset import EntailmentDataset, Sample

EVAL_BSIZE = 32

BEST_MODEL_CHECKPOINT_FNAME = 'best-model.pt'


def make_dataloader(dataset: EntailmentDataset, batch_size: int, split: str) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), collate_fn=lambda x: x)


class ModelInstructor:
    def __init__(self, model: torch.nn.Module, device, cfg_optimizer: Dict[str, Any], run_folder: Optional[str] = None):
        self.model = model
        self.device = device
        self.cfg_optimizer = cfg_optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_optimizer['learning_rate'])
        self.tracked_metrics = None
        self.best_dev_metrics = None
        self.run_folder = None
        self._refresh_tracked_metrics()
        if not run_folder:
            self._make_run_folder()

    def _refresh_tracked_metrics(self):
        self.tracked_metrics = {metric: [] for metric in ['loss', 'acc']}

    def _make_run_folder(self):
        if not os.path.exists('results'):
            os.mkdir('results')
        now = datetime.now()
        datestring = now.strftime("%b-%d-%Y %H.%M")
        self.run_folder = os.path.join('results', datestring)
        os.mkdir(self.run_folder)

    def calc_metrics(self, batch: Iterable[Sample], batch_predictions, batch_labels) -> Tuple[torch.Tensor, Dict[str, float]]:
        bsize = batch_predictions.shape[0]
        batch_weights = torch.tensor([s.sample_weight for s in batch]).to(self.device)
        criterion = torch.nn.BCELoss()
        # criterion = torch.nn.CrossEntropyLoss(reduction='none')
        # loss = criterion(batch_predictions, batch_labels)
        loss = criterion(batch_predictions[:, 1], batch_labels[:, 1])
        loss = (loss * batch_weights / batch_weights.sum()).sum()

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

    def _handle_dev_checkpoint(self, dev_metrics) -> bool:
        should_save = False
        if self.best_dev_metrics is None:
            self.best_dev_metrics = dev_metrics
            should_save = True

        tracking_metric = self.cfg_optimizer['dev_tracking_metric']
        tracking_function = {
                'loss': operator.lt,
                'acc': operator.gt,
                'auc': operator.gt
        }[tracking_metric]

        if tracking_function(dev_metrics[tracking_metric], self.best_dev_metrics[tracking_metric]):
            should_save = True

        if should_save:
            self.best_dev_metrics = dev_metrics
            print(colors.GREEN + '* Saving as "best" model...' + colors.ENDC)
            torch.save(self.model.state_dict(), os.path.join(self.run_folder, BEST_MODEL_CHECKPOINT_FNAME))

        return should_save

    def process_batch(self, batch: Iterable[Sample]) -> Tuple[Any, Any, torch.tensor]:
        batch_premises = [sample.premise for sample in batch]
        # batch_premises = [premise_triple[1] for premise_triple in batch_premises]
        batch_premises = [' '.join(premise_triple) for premise_triple in batch_premises]

        batch_hypotheses = [sample.hypothesis for sample in batch]
        # batch_hypotheses = [hypothesis_triple[1] for hypothesis_triple in batch_hypotheses]
        batch_hypotheses = [' '.join(hypothesis_triple) for hypothesis_triple in batch_hypotheses]

        truth_values = [int(sample.truth_value) for sample in batch]
        batch_labels = torch.nn.functional.one_hot(torch.tensor(truth_values), num_classes=2).float().to(self.device)
        return batch_premises, batch_hypotheses, batch_labels

    def step(self, batch: Iterable[Sample], training=True) -> Tuple[Dict[str, float], torch.tensor, torch.tensor]:
        # extract data
        batch_premises, batch_hypotheses, batch_labels = self.process_batch(batch)

        # forward pass
        self.model.train()
        batch_predictions = self.model(batch_premises, batch_hypotheses)

        # eval results
        loss, batch_metrics = self.calc_metrics(batch, batch_predictions, batch_labels)

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

        return batch_metrics, batch_predictions, batch_labels

    def train_model(self, train_dataset: Optional[EntailmentDataset], dev_dataset: Optional[EntailmentDataset]) -> str:
        batch_size = self.cfg_optimizer['batch_size']
        train_data = make_dataloader(train_dataset, batch_size, 'train')

        max_train_steps = self.cfg_optimizer['max_train_steps']
        epoch = 0
        prev_best_epoch = 0
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

                if dev_dataset:
                    print()
                    dev_metrics, _, _ = self.eval_model(dev_dataset, 'dev')

                    # decide whether to save the model as current "best"
                    new_best = self._handle_dev_checkpoint(dev_metrics)
                    if new_best:
                        prev_best_epoch = epoch
                    else:
                        patience = self.cfg_optimizer['patience']
                        if epoch - prev_best_epoch >= patience:
                            print(colors.UNDERLINE + f'No dev improvement in {patience} epochs. Stopping!' + colors.ENDC)
                            break

        return self.run_folder

    def eval_model(self, dataset: EntailmentDataset, split: str) \
            -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        print(f'Evaluating {split}: {dataset.name}...')
        self.model.eval()
        eval_data = make_dataloader(dataset, EVAL_BSIZE, split)

        all_metrics = {}
        num_samples = 0

        all_predictions = torch.tensor([]).to(self.device)
        all_labels = torch.tensor([]).to(self.device)

        for batch_num, batch in enumerate(eval_data):
            this_batch_size = len(batch)
            num_samples += this_batch_size
            batch_metrics, batch_predictions, batch_labels = self.step(batch, training=False)
            all_predictions = torch.cat([all_predictions, batch_predictions.detach()])
            all_labels = torch.cat([all_labels, batch_labels.detach()])

            for metric in batch_metrics:
                new_metric_val = (batch_metrics[metric] * this_batch_size)
                all_metrics[metric] = all_metrics[metric] + new_metric_val if metric in all_metrics else new_metric_val

        for metric in all_metrics:
            all_metrics[metric] /= num_samples

        loss = all_metrics['loss']
        acc = all_metrics['acc']

        all_labels_idx = torch.argmax(all_labels, dim=1).detach().cpu().numpy()
        all_predictions_true = all_predictions[:, 1].detach().cpu().numpy()
        precisions, recalls, thresholds = metrics.precision_recall_curve(all_labels_idx, all_predictions_true)
        auc = metrics.auc(recalls, precisions)

        max_recall = np.max(recalls)
        random_baseline_prec = np.sum(all_labels_idx)/len(all_labels_idx)
        auc_norm = (auc - (random_baseline_prec * max_recall)) / (1 - (random_baseline_prec * 1))

        all_metrics['auc'] = auc
        all_metrics['auc_norm'] = auc_norm

        print_bar('-')
        print(f'{colors.BOLD}{split}{colors.ENDC} |'
              f'\tloss: {loss:.3f}'
              f'\tacc: {acc*100:.1f}'
              f'\t{colors.GREEN}auc: {auc*100:.2f}{colors.ENDC}'
              f'\t{colors.YELLOW}auc_norm: {auc_norm*100:.2f}{colors.ENDC}'
              f'\tclass prec: {random_baseline_prec*100:.2f}')
        print_bar('-')

        info = {
                'random_baseline_prec': random_baseline_prec,
                'len_dataset': len(all_labels_idx)
        }

        return all_metrics, (precisions, recalls), info

