samples = []

# entailment types (quoted from Sports repo: https://gitlab.com/lianeg/temporal-entailment-sports-dataset)
#   0 = non-entailment; P does not entail Q
#   1 = entailment; P entails Q
#  -1 = directional non-entailment, i.e. the set of entailments with direction reversed; P does not entail Q
#  -2 = paraphrase; P is a paraphrase of Q
keep = [1, -1]
out_fname = 'dir_s.txt'

with open('sports_entailment.txt') as file:
	for line in file:
		premise, hypothesis, value = line.split('\t')
		value = int(value)
		if value not in keep:
			continue

		p = ('Team A', f'did {premise}', 'Team B')
		h = ('Team A', f'did {hypothesis}', 'Team B')

		samples.append((p, h, 'True' if value in [1, -2] else 'False'))

with open(out_fname, 'w+') as f:
	for (p, h, t) in samples:
		f.write(','.join(h) + '\t' + ','.join(p) + '\t' + t + '\n')
