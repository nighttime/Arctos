from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HIDDEN_SIZE = 768
NULL_RELATION_TEXT = 'NULL RELATION'


class ProjectorModel(torch.nn.Module):
	def __init__(self, device, cfg_model: Dict[str, Any]):
		super(ProjectorModel, self).__init__()

		self.device = device
		self.cfg_model = cfg_model

		self.encoder = SentenceTransformer(cfg_model['encoder_model'], device=device)
		self.decoder = torch.nn.ModuleList([
				torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
				for _ in range(self.cfg_model['classifier_hidden_layers'])
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
							p=self.cfg_model['decoder_dropout']))

		# predicted_premises = torch.nn.functional.dropout(
		# 					torch.nn.functional.gelu(encoded_hypotheses),
		# 					p=self.cfg_model['decoder_dropout'])

		for i, layer in enumerate(self.decoder):
			predicted_premises = layer(predicted_premises)
			if i < len(self.decoder) - 1:
				predicted_premises = torch.tanh(predicted_premises)

		# shape: B
		dists_predictions2prems = -torch.nn.functional.pairwise_distance(predicted_premises, encoded_premises)

		# shape: B
		dists_predictions2null = -torch.cdist(predicted_premises, null_premise).squeeze(dim=1)

		# shape: B x 2
		choices_dists = torch.stack([dists_predictions2null, dists_predictions2prems], dim=1)
		t = self.cfg_model['softmax_temperature']  # if self.training else 1
		choices_probs = torch.nn.functional.softmax(choices_dists/t, dim=1)

		return choices_probs


class VecDiffModel(torch.nn.Module):
	def __init__(self, device, cfg_model: Dict[str, Any]):
		super(VecDiffModel, self).__init__()

		self.device = device
		self.cfg_model = cfg_model

		self.encoder = SentenceTransformer(cfg_model['encoder_model'], device=device)

		num_hlayers = self.cfg_model['classifier_hidden_layers']
		num_input_vecs = 3
		if num_hlayers == 1:
			self.decoder = torch.nn.ModuleList([torch.nn.Linear(HIDDEN_SIZE * num_input_vecs, 2)])
		else:
			self.decoder = torch.nn.ModuleList(
					[torch.nn.Linear(HIDDEN_SIZE + 1, HIDDEN_SIZE)] +
					[torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(num_hlayers - 2)] +
					[torch.nn.Linear(HIDDEN_SIZE, 2)]
			)
		self.encoder_layer_norm = torch.nn.LayerNorm(HIDDEN_SIZE)
		self.encoder_batch_norm = torch.nn.BatchNorm1d(HIDDEN_SIZE)
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

		# Get sentence embedding and finetune encoder
		tokens_prem = {name: tsr.to(self.device) for name, tsr in self.encoder.tokenize(premises).items()}
		encoded_premises = self.encoder(tokens_prem)['sentence_embedding']
		tokens_hyp = {name: tsr.to(self.device) for name, tsr in self.encoder.tokenize(hypotheses).items()}
		encoded_hypotheses = self.encoder(tokens_hyp)['sentence_embedding']

		# Get predicate embedding and finetune encoder
		# def get_prop_embs(props: List[Tuple[str, str, str]]):
		# 	props_in_context = [' '.join(prop) for prop in props]
		# 	props_pred_only = [prop[1] for prop in props]
		#
		# 	tokens_prop_in_context = self.encoder.tokenize(props_in_context)['input_ids']
		# 	tokens_prop = self.encoder.tokenize(props_pred_only)['input_ids']
		# 	start, end = [find_subsequence(tokens_prop[j], tokens_prop_in_context[j]) for j in range(len(tokens_prop))]
		#
		# 	ts = {name: tsr.to(self.device) for name, tsr in self.encoder.tokenize(props_in_context).items()}
		# 	embeddings_context = self.encoder(ts)['token_embeddings']
		# 	return embeddings_context
		#
		# encoded_premises = get_prop_embs(premises)
		# encoded_hypotheses = get_prop_embs(hypotheses)

		# shape: B x HIDDEN_SIZE
		vector_differences = encoded_premises - encoded_hypotheses

		# shape: B x (3 x HIDDEN_SIZE)
		# encoder_out = vector_differences
		# encoder_out = torch.cat([encoded_premises, encoded_hypotheses], dim=1)
		# encoder_out = torch.cat([encoded_premises, encoded_hypotheses, vector_differences], dim=1)
		# similarities = torch.cosine_similarity(encoded_premises, encoded_hypotheses).unsqueeze(1)
		vector_distances = torch.nn.functional.pairwise_distance(encoded_hypotheses, encoded_premises).unsqueeze(1)
		encoder_out = torch.cat([vector_distances, vector_differences], dim=1)

		# encoder_out = torch.nn.functional.gelu(encoder_out)
		encoder_out = torch.nn.functional.dropout(encoder_out, p=self.cfg_model['decoder_dropout'])
		# encoder_out = self.encoder_batch_norm(encoder_out)
		# encoder_out = self.encoder_layer_norm(encoder_out)

		# decoder_input = self.encoder_layer_norm(torch.nn.functional.dropout(torch.nn.functional.gelu(vector_differences), p=self.cfg_model['decoder_dropout']))

		layer_outputs = encoder_out
		for i, layer in enumerate(self.decoder):
			layer_outputs = layer(layer_outputs)
			if i < len(self.decoder) - 1:
				layer_outputs = torch.tanh(layer_outputs)
				# layer_outputs = torch.nn.functional.dropout(layer_outputs, p=self.cfg_model['decoder_dropout'])

		# shape: B x 2
		t = self.cfg_model['softmax_temperature']  # if self.training else 1
		choices_probs = torch.nn.functional.softmax(layer_outputs/t, dim=1)

		return choices_probs


def find_subsequence(needle: List[int], haystack: List[int]) -> Tuple[int, int]:
	for start in range(len(haystack)):
		end = start + len(needle)
		if haystack[start:end] == needle:
			return start, end
