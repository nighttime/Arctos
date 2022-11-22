from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
import torch
import pprint
from config import RANDOM_SEED, CFG_HYPERPARAMETERS, CFG_OPTIMIZER, CFG_DATASET
from dataset.entailment_dataset import load_datasets
from model.ent_model import EntailmentModel
from train_and_eval import ModelInstructor

ex = Experiment('train_model', interactive=True)
ex.add_config({'cfg_hyperparameters': CFG_HYPERPARAMETERS})
ex.add_config({'cfg_optimizer': CFG_OPTIMIZER})
ex.add_config({'cfg_dataset': CFG_DATASET})
ex.captured_out_filter = apply_backspaces_and_linefeeds


def setup() -> torch.device:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
    return device


@ex.capture
def train_and_eval(device, cfg_hyperparameters, cfg_optimizer, cfg_dataset):
    # Load dataset
    train_info = cfg_dataset['train']
    train_description, train, dev, test = load_datasets(train_info['name'], train_info['version'])
    print(train_description)

    test_suite = None
    if cfg_dataset['additional_test_suite']:
        test_info = cfg_dataset['test']
        test_description, _, _, test_suite = load_datasets(test_info['name'], test_info['version'])
        print(test_description)

    # Make a model
    model = EntailmentModel(device, cfg_hyperparameters)
    model.to(device)

    # Train the model
    instructor = ModelInstructor(model, device, cfg_optimizer)
    instructor.train_model(train, dev)

    # Evaluate the model
    if test_suite:
        for test_set in test_suite:
            instructor.eval_model(test_set, 'test')
    elif test:
        for test_set in test:
            instructor.eval_model(test_set, 'test')


@ex.automain
def main(cfg_hyperparameters, cfg_optimizer, cfg_dataset):
    print('*'*100)
    pprint.pprint(cfg_hyperparameters)
    pprint.pprint(cfg_optimizer)
    pprint.pprint(cfg_dataset)
    device = setup()
    print(f'Device: {device}')
    print('*'*100)
    train_and_eval(device)
