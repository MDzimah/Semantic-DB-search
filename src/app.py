"""Streamlit UI application."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

try:
	from .data_loader import ImportedDataset, apply_deciding_columns, load_excel_dataset, suggest_deciding_columns
	from .search_engine import SearchEngine
	from .state_store import append_history_entry, ensure_directories, load_history, load_settings, save_dataset, save_settings
except ImportError:  # pragma: no cover - direct script execution support
	from data_loader import ImportedDataset, apply_deciding_columns, load_excel_dataset, suggest_deciding_columns
	from search_engine import SearchEngine
	from state_store import append_history_entry, ensure_directories, load_history, load_settings, save_dataset, save_settings


APP_TITLE = "Semantic DB Search"
DEFAULT_RESULT_COUNT = 5


def initialize_state() -> None:
	ensure_directories()
	defaults = {
		"base_dataset": None,
		"dataset": None,
		"engine": None,
		"deciding_columns": [],
		"uploaded_file_name": None,
		"results_per_query": DEFAULT_RESULT_COUNT,
		"sort_mode": "recency",
		"collapse_all": False,
		"history": load_history(),
		"query_input": "",
		"status": "Upload an Excel file to begin.",
	}
	loaded_settings = load_settings()
	defaults.update({key: value for key, value in loaded_settings.items() if key in defaults})
	for key, value in defaults.items():
		st.session_state.setdefault(key, value)


def reset_session() -> None:
	st.session_state["base_dataset"] = None
	st.session_state["dataset"] = None
	st.session_state["engine"] = None
	st.session_state["deciding_columns"] = []
	st.session_state["query_input"] = ""
	st.session_state["status"] = "Upload an Excel file to begin."
	st.session_state["collapse_all"] = False
	st.session_state["uploaded_file_name"] = None
	st.session_state["history"] = []
	save_settings(
		{
			"results_per_query": st.session_state.get("results_per_query", DEFAULT_RESULT_COUNT),
			"sort_mode": st.session_state.get("sort_mode", "recency"),
			"collapse_all": False,
		}
	)


def load_dataset_into_session(uploaded_file: Any) -> None:
	base_dataset = load_excel_dataset(uploaded_file)
	suggested = suggest_deciding_columns(base_dataset.headers)
	dataset = apply_deciding_columns(base_dataset, suggested)
	st.session_state.base_dataset = base_dataset
	st.session_state.dataset = dataset
	st.session_state.deciding_columns = suggested
	st.session_state.status = f"Loaded {len(dataset.rows)} rows from {dataset.sheet_name}."
	save_dataset(dataset, suggested)
	save_settings(
		{
			"results_per_query": st.session_state.results_per_query,
			"sort_mode": st.session_state.sort_mode,
			"collapse_all": st.session_state.collapse_all,
		}
	)


def build_engine_if_needed() -> SearchEngine | None:
	dataset = st.session_state.get("dataset")
	if dataset is None:
		return None

	engine = st.session_state.get("engine")
	if engine is None:
		try:
			engine = SearchEngine()
		except RuntimeError as error:
			st.error(str(error))
			return None
		st.session_state.engine = engine

	current_columns = list(st.session_state.get("deciding_columns", []))
	if getattr(engine, "dataset", None) is not dataset or getattr(engine, "selected_deciding_columns", []) != current_columns:
		engine.build(dataset, current_columns)
	return engine


def store_history_for_query(query: str, matches: list[Any], dataset: ImportedDataset) -> None:
	serialized_matches = []
	for match in matches:
		serialized_matches.append(
			{
				"row_index": match.row_index,
				"score": match.score,
				"semantic_score": match.semantic_score,
				"decision_score": match.decision_score,
				"lexical_score": match.lexical_score,
				"row": match.row.values,
				"display_text": match.row.display_text,
			}
		)
	entry = append_history_entry(
		{
			"query": query,
			"results": serialized_matches,
			"dataset_fingerprint": dataset.fingerprint,
			"saved_at": datetime.now(timezone.utc).isoformat(),
		}
	)
	st.session_state.history = entry


def render_controls() -> Any:
	top_left, top_right = st.columns([1, 2])
	with top_left:
		st.session_state.results_per_query = st.number_input(
			"Results per query",
			min_value=1,
			max_value=50,
			value=int(st.session_state.get("results_per_query", DEFAULT_RESULT_COUNT)),
			step=1,
		)
		st.session_state.sort_mode = st.selectbox(
			"Sort saved queries",
			["recency", "alphabetical"],
			index=0 if st.session_state.get("sort_mode", "recency") == "recency" else 1,
		)
	with top_right:
		upload_col, reset_col = st.columns([3, 1])
		with upload_col:
			uploaded_file = st.file_uploader("Import Excel workbook", type=["xlsx", "xlsm"], label_visibility="visible")
		with reset_col:
			if st.button("Reset", use_container_width=True):
				reset_session()
				st.rerun()
	save_settings(
		{
			"results_per_query": st.session_state.results_per_query,
			"sort_mode": st.session_state.sort_mode,
			"collapse_all": st.session_state.collapse_all,
		}
	)
	return uploaded_file


def render_deciding_columns() -> None:
	base_dataset = st.session_state.get("base_dataset")
	if base_dataset is None:
		return

	st.subheader("Deciding columns")
	st.caption("Choose the columns that define identity, brokerage, issuer, ticker, ISIN, or family.")
	selected = st.multiselect(
		"",
		options=base_dataset.headers,
		default=st.session_state.get("deciding_columns", []),
		label_visibility="collapsed",
	)
	if selected != st.session_state.get("deciding_columns", []):
		st.session_state.deciding_columns = selected
		dataset = apply_deciding_columns(base_dataset, selected)
		st.session_state.dataset = dataset
		save_dataset(dataset, selected)
		save_settings(
			{
				"results_per_query": st.session_state.results_per_query,
				"sort_mode": st.session_state.sort_mode,
				"collapse_all": st.session_state.collapse_all,
			}
		)
		st.session_state.engine = None


def render_history() -> list[dict[str, Any]]:
	history = list(st.session_state.get("history", []))
	if not history:
		st.info("No saved queries yet.")
		return []

	if st.session_state.get("sort_mode", "recency") == "alphabetical":
		history.sort(key=lambda item: item.get("query", "").lower())
	else:
		history.sort(key=lambda item: item.get("saved_at", ""), reverse=True)
	return history


def render_results(history: list[dict[str, Any]]) -> None:
	if not history:
		return

	st.checkbox("Collapse all", key="collapse_all")
	for item in history:
		query = item.get("query", "")
		results = item.get("results", [])
		expanded = not st.session_state.get("collapse_all", False)
		with st.expander(f"{query} ({len(results)} results)", expanded=expanded):
			for rank, result in enumerate(results, start=1):
				score = result.get("score", 0.0)
				row = result.get("row", {})
				display_text = result.get("display_text", "")
				st.markdown(f"**{rank}. Score:** {score:.3f}")
				st.write(display_text or row)
				st.divider()


def render_query_form(engine: SearchEngine | None) -> None:
	dataset = st.session_state.get("dataset")
	if dataset is None or engine is None:
		return

	st.subheader("Paste queries")
	with st.form("query_form", clear_on_submit=False):
		query_text = st.text_area(
			"One query per line",
			value=st.session_state.get("query_input", ""),
			height=180,
			placeholder="Paste one fund query per line",
			label_visibility="collapsed",
		)
		submitted = st.form_submit_button("Search")

	st.session_state.query_input = query_text

	if not submitted:
		return

	queries = [line.strip() for line in query_text.splitlines() if line.strip()]
	if not queries:
		st.warning("Paste at least one query.")
		return

	for query in queries:
		matches = engine.search(query, top_k=int(st.session_state.results_per_query))
		store_history_for_query(query, matches, dataset)

	st.session_state.status = f"Processed {len(queries)} query rows."
	st.rerun()


def main() -> None:
	st.set_page_config(page_title=APP_TITLE, layout="wide")
	initialize_state()

	st.title(APP_TITLE)
	st.caption("Local hybrid semantic search over Excel data.")

	uploaded_file = render_controls()
	if uploaded_file is not None and (
		st.session_state.get("dataset") is None
		or st.session_state.get("uploaded_file_name") != getattr(uploaded_file, "name", None)
	):
		load_dataset_into_session(uploaded_file)
		st.session_state.uploaded_file_name = getattr(uploaded_file, "name", None)

	dataset = st.session_state.get("dataset")
	if dataset is None:
		st.info(st.session_state.get("status", "Upload an Excel file to begin."))
		return

	render_deciding_columns()

	engine = build_engine_if_needed()
	if engine is None:
		st.warning("Select deciding columns to build the search index.")
		return

	st.success(st.session_state.get("status", "Ready."))

	history = render_history()
	render_results(history)
	render_query_form(engine)


if __name__ == "__main__":
	main()
