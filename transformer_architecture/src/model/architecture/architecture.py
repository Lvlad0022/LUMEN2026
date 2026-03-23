"""Configurable sequence-to-sequence Transformer model."""

from __future__ import annotations

import torch
from torch import nn

'''

class Transformer(nn.Module):
	"""Seq2Seq Transformer with configurable depth/width/scaling/output size."""

	def __init__(
		self,
		vocab_size: int = 32,
		d_model: int | None = None,
		num_layers: int | None = None,
		scale_factor: float = 4.0,
		num_outputs: int | None = None,
		nhead: int = 4,
		dropout: float = 0.1,
		max_seq_len: int = 512,
		# Backward-compatible aliases with older config naming
		**kwargs,
	) -> None:
		super().__init__()

		self.d_model = d_model if d_model is not None else 128
		self.num_layers = num_layers if num_layers is not None else 3
		self.scale_factor = scale_factor
		self.vocab_size = vocab_size
		self.num_outputs = num_outputs if num_outputs is not None else vocab_size

		ff_dim = max(int(self.d_model * self.scale_factor), self.d_model + 1)

		self.src_embedding = TokenTimeEmbedding(
			vocab_size=vocab_size,
			width=self.d_model,
			max_seq_len=max_seq_len,
		)
		self.tgt_embedding = TokenTimeEmbedding(
			vocab_size=vocab_size,
			width=self.d_model,
			max_seq_len=max_seq_len,
		)

		# depth = number of full Transformer blocks, each with exactly one encoder and one decoder
		self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
		self.transformer_block = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_layers)

		self.output_projection = nn.Linear(self.d_model, self.num_outputs)

	def forward(
		self,
		src_tokens: torch.Tensor,
		tgt_tokens: torch.Tensor,
		src_time: torch.Tensor | None = None,
		tgt_time: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Return logits of shape ``(batch, target_seq_len, num_outputs)``."""
		src = self.src_embedding(src_tokens, src_time)
		tgt = self.tgt_embedding(tgt_tokens, tgt_time)

		tgt_mask = self._causal_mask(tgt_tokens.size(1), tgt_tokens.device)
		src = self.transformer_block(src, tgt_mask=tgt_mask)

		return self.output_projection(tgt)

	@staticmethod
	def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
		"""Create an additive mask that prevents attention to future tokens."""
		mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
		return mask.masked_fill(mask == 1, float("-inf"))
	
'''
	
class CNN(nn.Module):
	"""Simple CNN for handwritten digit classification (e.g., MNIST)."""

	def __init__(
		self,
		in_channels: int = 1,
		num_classes: int = 10,
		dropout: float = 0.25,
		**kwargs,
	) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14
			nn.Dropout(dropout),
			
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),  # 14x14 -> 7x7
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(128 * 7 * 7, 256),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(256, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return logits of shape ``(batch_size, num_classes)``."""
		x = self.features(x)
		return self.classifier(x)