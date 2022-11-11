from torch.utils.data import Dataset
import json
from typing import Tuple, List, Optional, TypeVar
from dataclasses import dataclass
import os

from utils import DATASET_LOCATION


@dataclass
class Sample:
    hypothesis: Tuple[str, str, str]
    premise: Tuple[str, str, str]
    truth_value: bool
    flipped_args: bool


TEntDataset = TypeVar('TEntDataset', bound=Dataset)


class EntailmentDataset(Dataset):
    def __init__(self, file_path: str):
        self._samples = []
        with open(file_path, 'r') as file:
            for json_str in file:
                sample = Sample(**json.loads(json_str))
                self._samples.append(sample)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    @classmethod
    def from_found_file(cls, name: str, version: str, split: str) -> Optional[TEntDataset]:
        folder_name = f'{name}_v{version}'
        file_name = f'{split}.jsonl'
        file_path = os.path.join(DATASET_LOCATION, folder_name, file_name)
        try:
            return EntailmentDataset(file_path)
        except FileNotFoundError:
            return None


def load_datasets(name: str, version: str) -> \
        Tuple[Optional[EntailmentDataset], Optional[EntailmentDataset], Optional[EntailmentDataset]]:

    train = EntailmentDataset.from_found_file(name, version, 'train')
    dev = EntailmentDataset.from_found_file(name, version, 'dev')
    test = EntailmentDataset.from_found_file(name, version, 'test')
    return train, dev, test
