from typing import List, Dict, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HIDDEN_SIZE = 768
NULL_RELATION_TEXT = 'NULL RELATION'


class EntailmentModel(torch.nn.Module):
	def __init__(self, cfg_hyperparameters):
		super(EntailmentModel, self).__init__()

		self.cfg_hyperparameters = cfg_hyperparameters

		self.encoder = SentenceTransformer(cfg_hyperparameters['encoder_model'])
		self.decoder = torch.nn.ModuleList([
				torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
				for _ in range(self.cfg_hyperparameters['classifier_hidden_layers'])
		])
		self.encoder_layer_norm = torch.nn.LayerNorm(HIDDEN_SIZE)
		self._init_weights()

	def _init_weights(self):
		self.encoder_layer_norm.bias.data.zero_()
		self.encoder_layer_norm.weight.data.fill_(1.0)

		for layer in self.decoder:
			layer.weight.data.normal_()
			layer.bias.data.zero_()

	def forward(self, premises: List[str], hypotheses: List[str]) -> torch.Tensor:
		# shape: B x HIDDEN_SIZE
		encoded_premises = self.encoder.encode(premises, convert_to_tensor=True, show_progress_bar=False)
		encoded_hypotheses = self.encoder.encode(hypotheses, convert_to_tensor=True, show_progress_bar=False)

		# shape: 1 x HIDDEN_SIZE
		null_premise = self.encoder.encode([NULL_RELATION_TEXT], convert_to_tensor=True, show_progress_bar=False)

		# shape: B x HIDDEN_SIZE
		predicted_premises = \
			self.encoder_layer_norm(
				torch.nn.functional.dropout(
						torch.nn.functional.gelu(encoded_hypotheses),
						p=self.cfg_hyperparameters['decoder_dropout']))

		for layer in self.decoder:
			predicted_premises = layer(predicted_premises)

		# shape: B
		dists_predictions2prems = -torch.nn.functional.pairwise_distance(predicted_premises, encoded_premises)

		# shape: B
		dists_predictions2null = -torch.cdist(predicted_premises, null_premise).squeeze()

		# shape: B x 2
		choices_dists = torch.stack([dists_predictions2null, dists_predictions2prems], dim=1)
		choices_probs = torch.nn.functional.softmax(choices_dists, dim=1)

		return choices_probs
