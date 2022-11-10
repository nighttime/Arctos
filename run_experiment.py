from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
import torch
from config import RANDOM_SEED, CFG_HYPERPARAMETERS, CFG_OPTIMIZER, CFG_DATASET
from dataset.entailment_dataset import load_datasets
from model.ent_model import EntailmentModel
from train_and_eval import ModelInstructor

ex = Experiment('train_model', interactive=True)
ex.add_config({'cfg_hyperparameters': CFG_HYPERPARAMETERS})
ex.add_config({'cfg_optimizer': CFG_OPTIMIZER})
ex.add_config({'cfg_dataset': CFG_DATASET})
ex.captured_out_filter = apply_backspaces_and_linefeeds


def setup():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # TODO: set up device to run on GPU


@ex.capture
def train_and_eval(cfg_hyperparameters, cfg_optimizer, cfg_dataset):
    # Load dataset
    train, dev, test = load_datasets(cfg_dataset['name'], cfg_dataset['version'])
    datasets = {
            'train': train,
            'dev': dev,
            'test': test
    }

    # Make a model
    model = EntailmentModel(cfg_hyperparameters)

    # Train the model
    instructor = ModelInstructor(model, cfg_optimizer)
    instructor.train_model(datasets)

    # Evaluate the model


@ex.automain
def main(cfg_hyperparameters, cfg_optimizer):
    print(cfg_hyperparameters)
    print(cfg_optimizer)
    setup()
    train_and_eval()
