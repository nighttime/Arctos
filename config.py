from dataset import dataset_codes

RANDOM_SEED = 3

CFG_HYPERPARAMETERS = {
		'encoder_model': 'sentence-transformers/stsb-roberta-base-v2',
		'classifier_hidden_layers': 1,
		'decoder_dropout': 0.5,
}
CFG_OPTIMIZER = {
		'learning_rate': 1e-4,
		'batch_size': 64,
		'max_train_steps': 200
}
CFG_DATASET = {
	'train': {
		'name': dataset_codes.ANT_FULL,
		'version': 1
	},
	'additional_test_suite': False,
	'test': {
		'name': dataset_codes.TESTSUITE,
		'version': 1
	}
}
