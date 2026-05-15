"""Streamlit front-end for the Agent Forge multi-agent CSV analysis tool."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st

from src.agent.orchestrator import run_analysis_phase, run_clarify_phase
from src.config import settings
from src.logging_config import configure_logging
from src.memory.session_store import SessionStore
from src.schemas.models import AgentState

configure_logging(settings.log_level)

st.set_page_config(page_title="Agent Forge — Data Analysis", page_icon="🔬", layout="wide")
st.title("Agent Forge — Data Analysis Agent")
st.caption("Multi-agent CSV analysis powered by GPT-4o and LangGraph")

store = SessionStore(db_path=settings.db_path)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "phase" not in st.session_state:
    st.session_state["phase"] = "upload"
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = None

session_id: str = st.session_state["session_id"]

with st.sidebar:
    st.header("Session")
    st.code(session_id[:8] + "...", language=None)
    if st.button("New Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.divider()
    past = store.list_sessions()
    if past:
        st.subheader("Past Sessions")
        for sid in past[:5]:
            if st.button(sid[:12] + "...", key=f"load_{sid}"):
                loaded = store.load(sid)
                if loaded:
                    st.session_state["session_id"] = sid
                    st.session_state["agent_state"] = loaded
                    st.session_state["phase"] = "done" if loaded.final_response else "upload"
                    st.rerun()

phase: str = st.session_state["phase"]

if phase == "upload":
    st.subheader("Step 1: Upload your CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    user_query = st.text_input("What would you like to analyze?", value="Please analyze this dataset.")

    if uploaded and st.button("Analyze", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read())
            csv_path = tmp.name

        with st.spinner("Loading CSV and generating clarifying questions..."):
            try:
                state = run_clarify_phase(
                    csv_path=csv_path,
                    session_id=session_id,
                    user_query=user_query,
                )
                store.save(state)
                st.session_state["agent_state"] = state
                st.session_state["phase"] = "clarify" if state.clarifying_questions else "analyze"
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load CSV: {exc}")

elif phase == "clarify":
    state: AgentState = st.session_state["agent_state"]
    st.subheader("Step 2: Answer a few questions")

    if state.dataframe_profile:
        col1, col2 = st.columns(2)
        col1.metric("Rows", state.dataframe_profile.shape[0])
        col2.metric("Columns", state.dataframe_profile.shape[1])

    answers: dict[str, str] = {}
    for i, q in enumerate(state.clarifying_questions):
        answers[str(i)] = st.text_input(f"Q{i + 1}: {q}", key=f"q_{i}")

    if st.button("Run Analysis", type="primary"):
        ans_list = [answers.get(str(i), "") for i in range(len(state.clarifying_questions))]
        st.session_state["pending_answers"] = ans_list
        st.session_state["phase"] = "analyze"
        st.rerun()

elif phase == "analyze":
    state = st.session_state["agent_state"]
    answers_list: list[str] = st.session_state.get("pending_answers", [])

    with st.spinner("Running multi-agent analysis… this may take 30–90 seconds"):
        try:
            final_state = run_analysis_phase(state, answers=answers_list)
            store.save(final_state)
            st.session_state["agent_state"] = final_state
            st.session_state["phase"] = "done"
            st.rerun()
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.session_state["phase"] = "upload"

elif phase == "done":
    state = st.session_state["agent_state"]
    st.subheader("Analysis Results")

    if state.error:
        st.warning(f"Note: {state.error}")

    if state.final_response:
        st.markdown(state.final_response)

    all_charts = [c for r in state.subtask_results for c in r.charts]
    if all_charts:
        st.subheader("Charts")
        cols = st.columns(min(len(all_charts), 2))
        for i, chart_path in enumerate(all_charts):
            if Path(chart_path).exists():
                cols[i % 2].image(chart_path)

    with st.expander("Reasoning Trace"):
        for msg in state.messages:
            role_label = msg.role.value.upper()
            node = f" [{msg.node_name}]" if msg.node_name else ""
            st.markdown(f"**{role_label}{node}:** {msg.content}")

    with st.expander("Subtask Details"):
        for r in state.subtask_results:
            status = "✅" if r.success else "❌"
            st.markdown(f"{status} **{r.agent_type.value}** (subtask {r.subtask_id})")
            if r.code:
                st.code(r.code, language="python")
            if r.output:
                st.text(r.output[:1000])

    if st.button("Start New Analysis"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
