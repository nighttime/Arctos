import copy
import json
import os.path
import csv
import random
from typing import Tuple, List
from dataclasses import asdict
from collections import Counter

import dataset_codes as ds
from dataset.entailment_dataset import Sample

# Set file pointers to the chosen dataset
DS_MAIN = ds.LEVY_HOLT_DIR_DEV
DS_DEV_SET_PCT = 0.1
DS_DEV = None
DS_TEST = [ds.LEVY_HOLT_DIR_TEST, ds.LEVY_HOLT_PARAPHRASE_UNRELATED_TEST, ds.ANT_DIR, ds.ANT_FULL, ds.SPORTS_DIR]

# add random pairs in training with negative labels to better learn to discriminate unrelated predicates
AUG_ADD_RANDOM_NEGATIVE_PAIRS = 0
AUG_ADD_DS = ds.LEVY_HOLT_PARAPHRASE_UNRELATED_TRAIN

# Options: None, 'truth_value', 'origin_dataset'
WEIGHT_SAMPLES_BY_CLASS = 'truth_value'

assert not (DS_DEV_SET_PCT and DS_DEV)

file_paths_in = {
    ds.ANT_DIR: os.path.join('original', 'ant_in_levy_format', 'ant_directional_s2.txt'),
    ds.ANT_FULL: os.path.join('original', 'ant_in_levy_format', 'ant_full_s2.txt'),

    ds.LEVY_HOLT_DIR_DEV: os.path.join('original', 'levy_holt', 'dev_dir_s2.txt'),
    ds.LEVY_HOLT_DIR_TEST: os.path.join('original', 'levy_holt', 'test_dir_s2.txt'),
    ds.LEVY_HOLT_FULL_DEV: os.path.join('original', 'levy_holt', 'dev_s2.txt'),
    ds.LEVY_HOLT_FULL_TEST: os.path.join('original', 'levy_holt', 'test_s2.txt'),

    ds.LEVY_HOLT_PARAPHRASE_UNRELATED_TRAIN: os.path.join('original', 'LevyHoltMesh', 'Paraphrases_Unrelated', 'Full', 'train.txt'),
    ds.LEVY_HOLT_PARAPHRASE_UNRELATED_DEV: os.path.join('original', 'LevyHoltMesh', 'Paraphrases_Unrelated', 'Full', 'dev.txt'),
    ds.LEVY_HOLT_PARAPHRASE_UNRELATED_TEST: os.path.join('original', 'LevyHoltMesh', 'Paraphrases_Unrelated', 'Full', 'test.txt'),

    ds.SPORTS_DIR: os.path.join('original', 'sports', 'dir_s.txt')
}


def make_folder() -> str:
    folder_out_prefix = DS_MAIN or ds.TESTSUITE
    number_ds_exist = len([x for x in os.listdir('formatted') if x.startswith(folder_out_prefix)])
    folder_out_name = f'{folder_out_prefix}_v{number_ds_exist + 1}'
    folder_out = os.path.join('formatted', folder_out_name)
    os.makedirs(folder_out, exist_ok=True)
    print(f'Making {folder_out_name}...')
    return folder_out


def _process_line(hypo: str, prem: str, tval: str, origin_dataset) -> Sample:
    h_a0, h_pred, h_a1 = hypo.lower().split(',')
    p_a0, p_pred, p_a1 = prem.lower().split(',')

    # assume at least one argument will exactly match between premise and hypothesis; copy the other
    if h_a0 == p_a0:
        h_a1 = p_a1
    elif h_a1 == p_a0:
        h_a0 = p_a1

    elif h_a1 == p_a1:
        h_a0 = p_a0
    elif h_a0 == p_a1:
        h_a1 = p_a0

    return Sample(premise=(p_a0, p_pred, p_a1), hypothesis=(h_a0, h_pred, h_a1), truth_value=(tval == 'True'),
                  flipped_args=(p_a0 == h_a1), origin_dataset=origin_dataset)


def write_samples(samples, file):
    for sample in samples:
        file.write(json.dumps(asdict(sample)) + '\n')


def make_train(folder: str) -> Tuple[List[Sample], List[Sample]]:
    file_path = file_paths_in[DS_MAIN]
    if DS_DEV:
        file_path_dev = file_paths_in[DS_DEV]

    cleaned_dataset_samples = []
    with open(file_path, 'r') as file:
        csv_file = csv.reader(file, delimiter='\t')
        for line in csv_file:
            h, p, t = line[:3]
            sample = _process_line(h, p, t, DS_MAIN)
            cleaned_dataset_samples.append(sample)

    if DS_DEV_SET_PCT:
        sample_list = list(cleaned_dataset_samples)
        picks = [sample_list.pop(int(random.random() * len(sample_list)))]
        dev_sample_list = [picks[0]]
        while picks and len(dev_sample_list) / (len(cleaned_dataset_samples)) < DS_DEV_SET_PCT:
            current = picks.pop()
            current_preds = {current.premise[1], current.hypothesis[1]}

            adjacent_samples = []
            _sample_list = []
            for s in sample_list:
                preds = {s.premise[1], s.hypothesis[1]}
                if current_preds.intersection(preds):
                    adjacent_samples.append(s)
                else:
                    _sample_list.append(s)

            sample_list = _sample_list
            picks.extend(adjacent_samples)
            dev_sample_list.extend(adjacent_samples)

        assert len(sample_list) + len(dev_sample_list) == len(cleaned_dataset_samples), \
            f'{len(sample_list)} + {len(dev_sample_list)} != {len(cleaned_dataset_samples)}'

        train_preds = {p for sample_trip in sample_list for p in [sample_trip.premise[1], sample_trip.hypothesis[1]]}
        dev_preds = {p for sample_trip in dev_sample_list for p in [sample_trip.premise[1], sample_trip.hypothesis[1]]}

        assert len(train_preds.intersection(dev_preds)) == 0

        if len(sample_list) < len(dev_sample_list):
            sample_list, dev_sample_list = dev_sample_list, sample_list

        cleaned_dataset_samples = sample_list
        cleaned_dev_dataset_samples = dev_sample_list

    samples_main = cleaned_dataset_samples

    samples_dev = []
    if DS_DEV_SET_PCT:
        samples_dev = cleaned_dev_dataset_samples
    elif DS_DEV:
        with open(file_path_dev, 'r') as file:
            csv_file = csv.reader(file, delimiter='\t')
            for line in csv_file:
                h, p, t = line[:3]
                samples_dev.append(_process_line(h, p, t, DS_DEV))

    if AUG_ADD_RANDOM_NEGATIVE_PAIRS:
        samples_main = augment_with_random_unrelated(samples_main)

    if AUG_ADD_DS:
        samples_main = augment_with_dataset(samples_main)

    if WEIGHT_SAMPLES_BY_CLASS:
        assign_sample_weights(samples_main, WEIGHT_SAMPLES_BY_CLASS)

    if samples_dev:
        main_dump = 'train'
    else:
        main_dump = 'test'

    print(f'{main_dump}: {len(cleaned_dataset_samples)} augmented to-> {len(samples_main)}')
    with open(os.path.join(folder, f'{main_dump}.jsonl'), 'w') as file:
        write_samples(samples_main, file)

    if samples_dev:
        print(f'dev: {len(samples_dev)}')
        with open(os.path.join(folder, 'dev.jsonl'), 'w') as file:
            write_samples(samples_dev, file)

        return samples_main, samples_dev


def make_test(folder: str) -> List[List[Sample]]:
    samples_test = []
    for test_set in DS_TEST:
        samples_test_set = []
        file_path_test = file_paths_in[test_set]
        with open(file_path_test, 'r') as file:
            csv_file = csv.reader(file, delimiter='\t')
            for line in csv_file:
                h, p, t = line[:3]
                samples_test_set.append(_process_line(h, p, t, test_set))
        print(f'test [{test_set}]: {len(samples_test_set)}')
        with open(os.path.join(folder, f'test_{test_set}.jsonl'), 'w') as file:
            write_samples(samples_test_set, file)
        samples_test.append(samples_test_set)

    return samples_test


def mix_aug_samples(samples, aug_samples):
    new_samples = samples + aug_samples
    random.shuffle(new_samples)
    return new_samples


def augment_with_random_unrelated(dataset: List[Sample]) -> List[Sample]:
    target_amount = int(AUG_ADD_RANDOM_NEGATIVE_PAIRS*len(dataset))
    aug_samples = []
    for _ in range(target_amount):
        s1, s2 = random.sample(dataset, 2)
        new_prem = copy.deepcopy(s1.premise)
        new_hypo = (new_prem[0], s2.hypothesis[1], new_prem[2])
        if flipped := random.random() < 0.1:
            new_prem = (new_prem[2], new_prem[1], new_prem[0])
        new_sample = Sample(new_hypo, new_prem, False, flipped, ds.AUG_DATA)
        aug_samples.append(new_sample)

    return mix_aug_samples(dataset, aug_samples)


def augment_with_dataset(dataset: List[Sample]) -> List[Sample]:
    file_path_aug = file_paths_in[AUG_ADD_DS]
    aug_samples = []
    with open(file_path_aug, 'r') as file:
        csv_file = csv.reader(file, delimiter='\t')
        for line in csv_file:
            h, p, t = line[:3]
            aug_samples.append(_process_line(h, p, t, AUG_ADD_DS))

    return mix_aug_samples(dataset, aug_samples)


def assign_sample_weights(samples: List[Sample], attr: str):
    # Assign inverted class weights
    ctr = Counter()
    for s in samples:
        ctr[getattr(s, attr)] += 1
    total = len(samples)
    for s in samples:
        s.sample_weight = total / ctr[getattr(s, attr)]


def make_description(folder: str, train: List[Sample], dev: List[Sample], test: List[List[Sample]]):
    description = ''
    if DS_MAIN:
        description += f'TRAIN: {DS_MAIN}: {len(train)} (AUG: {AUG_ADD_RANDOM_NEGATIVE_PAIRS or AUG_ADD_DS})\n' \
                       f'DEV: {DS_DEV or DS_DEV_SET_PCT}: {len(dev)}\n'
        description += f'SAMPLE_WEIGHTING: {WEIGHT_SAMPLES_BY_CLASS}\n\n'
    if DS_TEST:
        description += f'TEST:\n'
        for name, samples in zip(DS_TEST, test):
            description += f'\t{name}: {len(samples)}\n'

    with open(os.path.join(folder, f'_description.txt'), 'w') as file:
        file.write(description)

    return description


def main():
    folder = make_folder()

    train_samples, dev_samples, test_samples = [], [], []

    if DS_MAIN:
        train_samples, dev_samples = make_train(folder)

    if DS_TEST:
        test_samples = make_test(folder)

    description = make_description(folder, train_samples, dev_samples, test_samples)

    print()
    print(f'done:')
    print(description)


if __name__ == '__main__':
    main()
