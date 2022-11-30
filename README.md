# Arctos
Learning entailment [premise |= hypothesis] by learning to search language model embedding space

## Model Architecture
This model consists of a sentence transformer which encodes both premise and hypothesis, plus an additional MLP which projects the hypothesis encoding somewhere else in embedding space. The model is trained to minimize the distance from this projection to the premise, and maximize the distance to a "null" premise.

## Installation
1. Clone the repo:

```
git clone git@github.com:nighttime/Arctos.git
```

2. Set up a conda environment using the bundled environment.yml file:

```
conda env create -f environment.yml
```

Note that if running on a machine with a GPU, you will need to install the gpu version of PyTorch and transformers separately.

## Usage
Within the conda environment, run the experiment:

```
python run_experiment.py [flags]
```

Arctos supports these flags: 
- d [allows debugging with sacred]
- a [cleanup after each run by deleting all model/results files]
- z [plot a mini precision-recall curve straight to the terminal for each dataset tested]

The -z command requires a separate install of the [uniplot](https://github.com/olavolav/uniplot) module:
```
pip install uniplot
```

On the first run, language model files will be downloaded and cached.

Experiment parameters can be changed in `config.py`

Generating new datasets can be done using the provided `prep_dataset.py` script, which can be configured with different train+dev configurations plus a suite of tests. These configurations are done using the global variables at the top of the file.

## The Name??
Arctos is the species name of the grizzly bear!
