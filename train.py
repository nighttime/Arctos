from typing import Dict, Any, Optional
from model.ent_model import EntailmentModel
from dataset.entailment_dataset import EntailmentDataset


def train_and_eval(model: EntailmentModel,
                   cfg_optimizer: Dict[str, Any],
                   datasets: Dict[str, Optional[EntailmentDataset]]):
    batch_size = cfg_optimizer['batch_size']

