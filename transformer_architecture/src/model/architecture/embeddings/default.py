"""Embedding layers for token + time encoding."""

from __future__ import annotations

import math

import torch
from torch import nn


class TimeEncoding(nn.Module):
	"""Sinusoidal encoding using timestamps instead of token positions."""

	def __init__(self, width: int, max_seq_len: int = 512) -> None:
		super().__init__()
		self.width = width
		self.max_seq_len = max_seq_len

		inv_freq = torch.exp(
			torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width)
		)
		self.register_buffer("inv_freq", inv_freq, persistent=False)

	def forward(self, x: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
		"""Add time encoding to embeddings.

		Args:
			x: Embeddings, shape ``(B, T, C)``.
			timestamps: Optional timestamp tensor, shape ``(B, T)`` (or ``(T,)``).
				If not provided, an increasing ``[0, 1, ..., T-1]`` timeline is used.
		"""
		batch_size, seq_len, _ = x.shape

		if timestamps is None:
			timestamps = torch.arange(seq_len, device=x.device, dtype=torch.float32)
			timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)
		elif timestamps.dim() == 1:
			timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)
		else:
			timestamps = timestamps.to(dtype=torch.float32)

		angles = timestamps.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
		enc = torch.zeros(batch_size, seq_len, self.width, device=x.device, dtype=x.dtype)
		enc[..., 0::2] = torch.sin(angles).to(dtype=x.dtype)
		enc[..., 1::2] = torch.cos(angles[..., : enc[..., 1::2].shape[-1]]).to(dtype=x.dtype)

		return x + enc


class TokenTimeEmbedding(nn.Module):
	"""Token embedding plus time encoding."""

	def __init__(self, vocab_size: int, width: int, max_seq_len: int = 512) -> None:
		super().__init__()
		self.width = width
		self.token_embedding = nn.Embedding(vocab_size, width)
		self.time_encoding = TimeEncoding(width=width, max_seq_len=max_seq_len)

	def forward(self, tokens: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
		embedded = self.token_embedding(tokens) * math.sqrt(self.width)
		return self.time_encoding(embedded, timestamps)
