"""Persistence helpers for datasets, search state, and query history."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np

try:
	from .data_loader import ImportedDataset
except ImportError:  # pragma: no cover - direct script execution support
	from data_loader import ImportedDataset

try:
	import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional fallback for development
	faiss = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "saved_state"
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"


def ensure_directories() -> None:
	for directory in (STATE_DIR, CACHE_DIR, DATA_DIR):
		directory.mkdir(parents=True, exist_ok=True)


def state_path(name: str) -> Path:
	ensure_directories()
	return STATE_DIR / name


def save_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Path, default: Any = None) -> Any:
	if not path.exists():
		return default
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def save_dataset(dataset: ImportedDataset, deciding_columns: list[str]) -> Path:
	payload = {
		"dataset": dataset.to_dict(),
		"deciding_columns": deciding_columns,
		"saved_at": datetime.now(timezone.utc).isoformat(),
	}
	path = state_path("dataset.json")
	save_json(path, payload)
	return path


def load_dataset() -> dict[str, Any] | None:
	return load_json(state_path("dataset.json"), default=None)


def save_settings(settings: dict[str, Any]) -> Path:
	path = state_path("settings.json")
	save_json(path, settings)
	return path


def load_settings() -> dict[str, Any]:
	return load_json(state_path("settings.json"), default={}) or {}


def save_history(history: list[dict[str, Any]]) -> Path:
	path = state_path("history.json")
	save_json(path, history)
	return path


def load_history() -> list[dict[str, Any]]:
	return load_json(state_path("history.json"), default=[]) or []


def save_faiss_index(index: Any, dimension: int | None) -> Path:
	path = state_path("index.faiss")
	if index is None:
		return path
	if faiss is None:
		np.save(state_path("index.npy"), index)
		save_json(state_path("index_meta.json"), {"dimension": dimension, "backend": "numpy"})
		return path
	faiss.write_index(index, str(path))
	save_json(state_path("index_meta.json"), {"dimension": dimension, "backend": "faiss"})
	return path


def load_faiss_index() -> Any:
	meta = load_json(state_path("index_meta.json"), default=None)
	if not meta:
		return None
	backend = meta.get("backend")
	if backend == "faiss" and faiss is not None:
		path = state_path("index.faiss")
		if path.exists():
			return faiss.read_index(str(path))
	if backend == "numpy":
		path = state_path("index.npy")
		if path.exists():
			return np.load(path, allow_pickle=True)
	return None


def save_query_result(query: str, results: list[dict[str, Any]], dataset_fingerprint: str) -> dict[str, Any]:
	return {
		"query": query,
		"results": results,
		"dataset_fingerprint": dataset_fingerprint,
		"saved_at": datetime.now(timezone.utc).isoformat(),
	}


def append_history_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
	history = load_history()
	history.append(entry)
	save_history(history)
	return history
