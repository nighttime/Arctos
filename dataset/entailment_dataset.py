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
    def __init__(self, file_path: str, name: str = None):
        if name is None:
            name = file_path
        self._samples = []
        self.name = name
        with open(file_path, 'r') as file:
            for json_str in file:
                sample = Sample(**json.loads(json_str))
                self._samples.append(sample)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    @classmethod
    def from_file(cls, file_path: str) -> Optional[TEntDataset]:
        try:
            return EntailmentDataset(file_path)
        except FileNotFoundError:
            return None


def load_datasets(name: str, version: str) -> \
        Tuple[str, Optional[EntailmentDataset], Optional[EntailmentDataset], Optional[List[EntailmentDataset]]]:

    splits = ['train', 'dev', 'test']
    datasets = dict.fromkeys(splits)
    folder_name = f'{name}_v{version}'
    folder_path = os.path.join(DATASET_LOCATION, folder_name)

    with open(os.path.join(folder_path, '_description.txt')) as desc_file:
        description_txt = desc_file.read()

    for split in splits:
        for file_name in os.listdir(folder_path):
            if file_name.startswith(split) and file_name.endswith('.jsonl'):
                file_path = os.path.join(folder_path, file_name)
                ds = EntailmentDataset(file_path, file_name)
                if split == 'test':
                    if ds and not datasets[split]:
                        datasets[split] = []
                    datasets[split].append(ds)
                else:
                    datasets[split] = ds

    return description_txt, datasets['train'], datasets['dev'], datasets['test']
