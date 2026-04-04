"""Search engine core logic.

This module embeds cleaned row text, indexes it with Faiss when available, and
reranks semantic candidates using deciding-column agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from rapidfuzz import fuzz

try:
	import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional fallback for development
	faiss = None

try:
	from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional fallback for development
	SentenceTransformer = None

try:
	from .data_loader import ImportedDataset, ImportedRow, normalize_text
except ImportError:  # pragma: no cover - direct script execution support
	from data_loader import ImportedDataset, ImportedRow, normalize_text


@dataclass(slots=True)
class SearchMatch:
	row_index: int
	score: float
	semantic_score: float
	decision_score: float
	lexical_score: float
	row: ImportedRow


class SearchEngine:
	"""Semantic search plus structured reranking."""

	def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
		self.model_name = model_name
		self.model = self._load_model(model_name)
		self.dataset: ImportedDataset | None = None
		self.documents: list[str] = []
		self.embeddings: np.ndarray | None = None
		self.index = None
		self.dimension: int | None = None
		self.selected_deciding_columns: list[str] = []

	def _load_model(self, model_name: str):
		if SentenceTransformer is None:
			raise RuntimeError(
				"sentence-transformers is not installed. Install the project dependencies first."
			)
		return SentenceTransformer(model_name)

	def build(self, dataset: ImportedDataset, deciding_columns: Iterable[str]) -> None:
		self.dataset = dataset
		self.selected_deciding_columns = [column for column in deciding_columns if column in dataset.headers]
		self.documents = [row.search_text for row in dataset.rows]

		if self.documents:
			self.embeddings = self.model.encode(
				self.documents,
				convert_to_numpy=True,
				normalize_embeddings=True,
				show_progress_bar=False,
			).astype(np.float32)
		else:
			self.embeddings = np.empty((0, 0), dtype=np.float32)

		if self.embeddings.size == 0:
			self.index = None
			self.dimension = None
			return

		self.dimension = int(self.embeddings.shape[1])
		self.index = self._build_index(self.embeddings)

	def _build_index(self, embeddings: np.ndarray):
		if faiss is None:
			return None
		index = faiss.IndexFlatIP(self.dimension)
		index.add(embeddings)
		return index

	def search(self, query: str, top_k: int = 10, candidate_pool: int | None = None) -> list[SearchMatch]:
		if not self.dataset or self.embeddings is None or len(self.dataset.rows) == 0:
			return []

		query_text = normalize_text(query)
		if not query_text:
			return []

		pool_size = candidate_pool or max(top_k * 10, 50)
		pool_size = min(pool_size, len(self.dataset.rows))

		query_embedding = self.model.encode(
			[query_text],
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		).astype(np.float32)

		if self.index is not None:
			distances, indices = self.index.search(query_embedding, pool_size)
			candidate_indices = [int(index) for index in indices[0] if index >= 0]
			semantic_lookup = {int(index): float(distance) for index, distance in zip(indices[0], distances[0]) if index >= 0}
		else:
			similarities = np.dot(self.embeddings, query_embedding[0])
			candidate_indices = np.argsort(similarities)[::-1][:pool_size].tolist()
			semantic_lookup = {int(index): float(similarities[index]) for index in candidate_indices}

		matches: list[SearchMatch] = []
		for index in candidate_indices:
			row = self.dataset.rows[index]
			semantic_score = semantic_lookup.get(index, 0.0)
			decision_score = self._decision_score(query_text, row)
			lexical_score = self._lexical_score(query_text, row)
			final_score = (semantic_score * 0.60) + (decision_score * 0.30) + (lexical_score * 0.10)
			matches.append(
				SearchMatch(
					row_index=row.row_index,
					score=final_score,
					semantic_score=semantic_score,
					decision_score=decision_score,
					lexical_score=lexical_score,
					row=row,
				)
			)

		matches.sort(key=lambda match: match.score, reverse=True)
		return matches[:top_k]

	def _decision_score(self, query_text: str, row: ImportedRow) -> float:
		if not self.selected_deciding_columns:
			return 0.0

		score = 0.0
		weight_total = 0.0
		for column in self.selected_deciding_columns:
			row_value = normalize_text(row.values.get(column, ""))
			if not row_value:
				continue

			column_weight = self._column_weight(column)
			weight_total += column_weight
			partial = fuzz.partial_ratio(query_text, row_value) / 100.0
			token = fuzz.token_set_ratio(query_text, row_value) / 100.0
			exact = 1.0 if query_text == row_value else 0.0
			score += column_weight * max(partial, token, exact)

		if weight_total == 0.0:
			return 0.0
		return score / weight_total

	def _lexical_score(self, query_text: str, row: ImportedRow) -> float:
		row_text = normalize_text(row.display_text)
		if not row_text:
			return 0.0
		return max(
			fuzz.partial_ratio(query_text, row_text) / 100.0,
			fuzz.token_set_ratio(query_text, row_text) / 100.0,
		)

	def _column_weight(self, column: str) -> float:
		normalized = normalize_text(column)
		if any(token in normalized for token in ("isin", "ticker")):
			return 1.6
		if any(token in normalized for token in ("broker", "issuer", "family", "provider", "manager")):
			return 1.35
		if "name" in normalized:
			return 1.15
		if any(token in normalized for token in ("asset", "category", "class")):
			return 1.05
		return 1.0
