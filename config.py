from dataset import dataset_codes

RANDOM_SEED = 3

MODEL_PROJECTOR = 'projector'
MODEL_VECDIFF = 'vecdiff'

CFG_MODEL = {
		'model_type': MODEL_PROJECTOR,
		'encoder_model': 'sentence-transformers/stsb-roberta-base-v2',
		# 'encoder_model': 'microsoft/deberta-v3-base',
		'classifier_hidden_layers': 2,
		'decoder_dropout': 0.25,
		'softmax_temperature': 1,
}
CFG_OPTIMIZER = {
		'learning_rate': 1e-4,
		'batch_size': 64,
		'max_train_steps': 2000,
		'dev_tracking_metric': 'auc',  # options are: (min) 'loss', (max) 'acc', 'auc'
		'patience': 10,
}
CFG_DATASET = {
	'train': {
		'name': dataset_codes.LEVY_HOLT_DIR_DEV,
		'version': 3
	},
	'additional_test_suite': False,
	'test': {
		'name': dataset_codes.TESTSUITE,
		'version': 2
	}
}
