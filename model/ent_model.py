from typing import List, Dict, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HIDDEN_SIZE = 768
NULL_RELATION_TEXT = 'NULL RELATION'


class EntailmentModel(torch.nn.Module):
	def __init__(self, device, cfg_hyperparameters: Dict[str, Any]):
		super(EntailmentModel, self).__init__()

		self.device = device
		self.cfg_hyperparameters = cfg_hyperparameters

		self.encoder = SentenceTransformer(cfg_hyperparameters['encoder_model'], device=device)
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
		# encoded_premises = self.encoder.encode(premises, convert_to_tensor=True, show_progress_bar=False)
		# encoded_hypotheses = self.encoder.encode(hypotheses, convert_to_tensor=True, show_progress_bar=False)

		# shape: 1 x HIDDEN_SIZE
		# null_premise = self.encoder.encode([NULL_RELATION_TEXT], convert_to_tensor=True, show_progress_bar=False)

		# alternative // keep the encoder network connected to the computation graph (finetune encoder)
		tokens_prem = {name: tsr.to(self.device) for name, tsr in self.encoder.tokenize(premises).items()}
		encoded_premises = self.encoder(tokens_prem)['sentence_embedding']
		tokens_hyp = {name: tsr.to(self.device) for name, tsr in self.encoder.tokenize(hypotheses).items()}
		encoded_hypotheses = self.encoder(tokens_hyp)['sentence_embedding']
		tokens_null = {name: tsr.to(self.device) for name, tsr in self.encoder.tokenize([NULL_RELATION_TEXT]).items()}
		null_premise = self.encoder(tokens_null)['sentence_embedding']

		# shape: B x HIDDEN_SIZE
		predicted_premises = \
			self.encoder_layer_norm(
				torch.nn.functional.dropout(
						torch.nn.functional.gelu(encoded_hypotheses),
						p=self.cfg_hyperparameters['decoder_dropout']))

		for i, layer in enumerate(self.decoder):
			predicted_premises = layer(predicted_premises)
			if i < len(self.decoder)-1:
				predicted_premises = torch.tanh(predicted_premises)
				# predicted_premises = torch.nn.functional.dropout(torch.tanh(predicted_premises), self.cfg_hyperparameters['decoder_dropout'])

		# shape: B
		dists_predictions2prems = -torch.nn.functional.pairwise_distance(predicted_premises, encoded_premises)

		# shape: B
		dists_predictions2null = -torch.cdist(predicted_premises, null_premise).squeeze(dim=1)

		# shape: B x 2
		choices_dists = torch.stack([dists_predictions2null, dists_predictions2prems], dim=1)
		choices_probs = torch.nn.functional.softmax(choices_dists, dim=1)

		return choices_probs
