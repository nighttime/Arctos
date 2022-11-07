from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from config import CFG_HYPERPARAMETERS, CFG_OPTIMIZER, CFG_DATASET
from dataset.entailment_dataset import load_datasets
from model.ent_model import EntailmentModel
from train import train_and_eval

ex = Experiment('train_model')
ex.add_config({'cfg_hyperparameters': CFG_HYPERPARAMETERS})
ex.add_config({'cfg_optimizer': CFG_OPTIMIZER})
ex.add_config({'cfg_dataset': CFG_DATASET})
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def train_model(cfg_hyperparameters, cfg_optimizer, cfg_dataset):
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
    train_and_eval(model, cfg_optimizer, datasets)

    # Evaluate the model


@ex.automain
def main(cfg_hyperparameters, cfg_optimizer):
    print(cfg_hyperparameters)
    print(cfg_optimizer)
    train_model()
