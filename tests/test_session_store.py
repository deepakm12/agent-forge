"""Tests for SessionStore CRUD operations: save, load, overwrite, list, and delete."""

from __future__ import annotations

from typing import Any

import pytest

from src.memory.session_store import SessionStore
from src.schemas.models import AgentState


@pytest.fixture
def store(tmp_path: Any) -> SessionStore:
    return SessionStore(db_path=str(tmp_path / "test.db"))


def test_save_and_load(store: SessionStore) -> None:
    state = AgentState(session_id="s1", csv_path="/tmp/f.csv", iteration=3)
    store.save(state)
    loaded = store.load("s1")
    assert loaded is not None
    assert loaded.session_id == "s1"
    assert loaded.iteration == 3


def test_load_missing_returns_none(store: SessionStore) -> None:
    assert store.load("nonexistent") is None


def test_overwrite_saves_latest(store: SessionStore) -> None:
    state = AgentState(session_id="s1", csv_path="/tmp/f.csv")
    store.save(state)
    updated = state.model_copy(update={"iteration": 5})
    store.save(updated)
    loaded = store.load("s1")
    assert loaded is not None
    assert loaded.iteration == 5


def test_list_sessions(store: SessionStore) -> None:
    store.save(AgentState(session_id="s1", csv_path="/tmp/a.csv"))
    store.save(AgentState(session_id="s2", csv_path="/tmp/b.csv"))
    ids = store.list_sessions()
    assert set(ids) == {"s1", "s2"}


def test_delete_session(store: SessionStore) -> None:
    store.save(AgentState(session_id="s1", csv_path="/tmp/f.csv"))
    store.delete("s1")
    assert store.load("s1") is None
