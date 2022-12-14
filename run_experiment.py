import os
import shutil
from typing import Dict, Any
import pdb

from sacred import Experiment, cli_option
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
import torch
import pprint
from utils import print_bar
from config import RANDOM_SEED, CFG_MODEL, CFG_OPTIMIZER, CFG_DATASET, MODEL_PROJECTOR, MODEL_VECDIFF
from dataset.entailment_dataset import load_datasets, EntailmentDataset, Sample
from model.ent_model import ProjectorModel, VecDiffModel
from train_and_eval import ModelInstructor, BEST_MODEL_CHECKPOINT_FNAME


@cli_option('-a', '--cleanup', is_flag=True)
def option_no_save_model(args, run):
    run.info['cleanup'] = args


@cli_option('-z', '--printcurve', is_flag=True)
def option_print_perf_curve_to_terminal(args, run):
    run.info['printcurve'] = args


ex = Experiment('train_model',
                additional_cli_options=[option_no_save_model, option_print_perf_curve_to_terminal],
                interactive=True)
ex.add_config({'cfg_model': CFG_MODEL})
ex.add_config({'cfg_optimizer': CFG_OPTIMIZER})
ex.add_config({'cfg_dataset': CFG_DATASET})
ex.captured_out_filter = apply_backspaces_and_linefeeds


def setup() -> torch.device:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
    return device


def plot_to_term(pr_results, info: Dict[str, Any]):
    import uniplot
    precisions, recalls = pr_results
    uniplot.plot(xs=recalls, ys=precisions, y_gridlines=[0, info['random_baseline_prec'], 1.0])
    # pdb.set_trace()


@ex.capture
def train_and_eval(device, cfg_model, cfg_optimizer, cfg_dataset):
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
    model_class = {
            MODEL_PROJECTOR: ProjectorModel,
            MODEL_VECDIFF: VecDiffModel,
    }[cfg_model['model_type']]

    model = model_class(device, cfg_model).to(device)

    # Train the model
    instructor = ModelInstructor(model, device, cfg_optimizer)
    run_folder = instructor.train_model(train, dev)

    # Evaluate the model
    print()
    print('Loading best model for eval...')
    best_model_path = os.path.join(run_folder, BEST_MODEL_CHECKPOINT_FNAME)
    best_model = model_class(device, cfg_model).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    evaluator = ModelInstructor(best_model, device, cfg_optimizer, run_folder=run_folder)
    if test_suite:
        for test_set in test_suite:
            _, pr_results, eval_info = evaluator.eval_model(test_set, 'test')
            if 'printcurve' in ex.info:
                plot_to_term(pr_results, eval_info)
    elif test:
        for test_set in test:
            _, pr_results, eval_info = evaluator.eval_model(test_set, 'test')
            if 'printcurve' in ex.info:
                plot_to_term(pr_results, eval_info)

    if 'cleanup' in ex.info and ex.info['cleanup']:
        shutil.rmtree(run_folder)


@ex.automain
def main(cfg_model, cfg_optimizer, cfg_dataset):
    print_bar()
    pprint.pprint(cfg_model)
    pprint.pprint(cfg_optimizer)
    pprint.pprint(cfg_dataset)
    device = setup()
    print(f'Device: {device}')
    print_bar()
    train_and_eval(device)
