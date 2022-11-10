RANDOM_SEED = 3

CFG_HYPERPARAMETERS = {
		'encoder_model': 'sentence-transformers/stsb-roberta-base-v2',
		'classifier_hidden_layers': 2,
		'decoder_dropout': 0.5,
}
CFG_OPTIMIZER = {
		'learning_rate': 1e-4,
		'batch_size': 64,
		'max_train_steps': 1000
}
CFG_DATASET = {
		'name': 'ANT-dir',
		'version': 2
}
