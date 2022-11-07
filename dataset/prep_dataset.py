import json
import os.path
import csv
import random
from typing import Tuple
from dataclasses import dataclass, asdict
import dataset_codes
from dataset.entailment_dataset import Sample

# Set file pointers to the chosen dataset
DS_MAIN = dataset_codes.ANT_DIR
DS_DEV_SET_PCT = 0.1
DS_DEV = None

file_paths_in = {
		dataset_codes.ANT_DIR: os.path.join('original', 'ant_in_levy_format', 'ant_directional_s2.txt'),
		dataset_codes.ANT_FULL: os.path.join('original', 'ant_in_levy_format', 'ant_full_s2.txt'),
}

file_path = file_paths_in[DS_MAIN]
if DS_DEV:
	file_path_dev = file_paths_in[DS_DEV]

number_ds_exist = len([x for x in os.listdir('formatted') if x.startswith(DS_MAIN)])
folder_out = os.path.join('formatted', f'{DS_MAIN}_v{number_ds_exist + 1}')
os.makedirs(folder_out, exist_ok=True)


@dataclass
class DatasetSample:
	premise: Tuple[str, str, str]
	hypothesis: Tuple[str, str, str]
	truth_value: bool
	flipped_args: bool


def _process_line(hypo: str, prem: str, tval: str) -> DatasetSample:
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

	return DatasetSample(premise=(p_a0, p_pred, p_a1),
						 hypothesis=(h_a0, h_pred, h_a1),
						 truth_value=(tval == 'True'),
						 flipped_args=(p_a0 == h_a1))


cleaned_dataset_samples = []
with open(file_path, 'r') as file:
	csv_file = csv.reader(file, delimiter='\t')
	for h, p, t in csv_file:
		dsample = _process_line(h, p, t)
		cleaned_dataset_samples.append(dsample)


if DS_DEV_SET_PCT:
	sample_list = list(cleaned_dataset_samples)
	picks = [sample_list.pop(int(random.random() * len(sample_list)))]
	dev_sample_list = [picks[0]]
	while picks or len(dev_sample_list) / (len(cleaned_dataset_samples)) < DS_DEV_SET_PCT:
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


def make_sample(dsample: DatasetSample):
	premises = [' '.join(dsample.premise)]
	hypothesis = ' '.join(dsample.hypothesis)

	return Sample(hypothesis=hypothesis,
					premises=premises,
					truth_value=dsample.truth_value)

samples_main = []
for dsample in cleaned_dataset_samples:
	samples_main.append(make_sample(dsample))

samples_dev = []
if DS_DEV_SET_PCT:
	for dsample in cleaned_dev_dataset_samples:
		samples_dev.append(make_sample(dsample))
elif DS_DEV:
	raise NotImplementedError('NYI')


def write_samples(samples, file):
	for sample in samples:
		file.write(json.dumps(asdict(sample)) + '\n')


if samples_dev:
	main_dump = 'train'
else:
	main_dump = 'test'

print(f'{main_dump}: {len(samples_main)}')
with open(os.path.join(folder_out, f'{main_dump}.jsonl'), 'w') as file:
	write_samples(samples_main, file)

if samples_dev:
	print(f'dev: {len(samples_dev)}')
	with open(os.path.join(folder_out, 'dev.jsonl'), 'w') as file:
		write_samples(samples_dev, file)

with open(os.path.join(folder_out, f'_description.txt'), 'w') as file:
	file.write(f'MAIN: {DS_MAIN} ({len(samples_main)})\nDEV: {DS_DEV or DS_DEV_SET_PCT} ({len(samples_dev)})')

print(f'done: {DS_MAIN} with {DS_DEV or DS_DEV_SET_PCT}')
