"""Streamlit dashboard for inspecting a Plasticity agent's local state.

Run it via ``plasticity dashboard`` (which shells out to ``streamlit run`` on
this file) or directly with ``streamlit run path/to/dashboard.py``. The memory
directory is read from the ``PLASTICITY_MEMORY_DIR`` environment variable.

This module is intentionally never imported by the package itself — it is a
standalone Streamlit script, so importing it requires the optional ``streamlit``
dependency only at run time.
"""

from __future__ import annotations

import os

import streamlit as st

from plasticity_agent.healing.detector import detect_failures
from plasticity_agent.healing.repair import heal
from plasticity_agent.memory.memory_os import MemoryOS
from plasticity_agent.reasoning.market import ReasoningMarket
from plasticity_agent.thermodynamics.energy_report import build_energy_report

MEMORY_DIR = os.environ.get("PLASTICITY_MEMORY_DIR", "./memory")


def _open() -> MemoryOS:
    return MemoryOS(memory_dir=MEMORY_DIR)


def render_overview(memory: MemoryOS) -> None:
    st.header("Overview")
    energy = build_energy_report(memory.list_memories(), memory.load_traces())
    col1, col2, col3 = st.columns(3)
    col1.metric("Memories", memory.count())
    col2.metric("Skills", memory.skills.count())
    col3.metric("Plasticity", f"{energy.plasticity_score:.0f}/100")
    st.write(energy.summary)


def render_memories(memory: MemoryOS) -> None:
    st.header("Memories")
    rows = [
        {
            "id": m.id,
            "type": m.memory_type,
            "content": m.content[:100],
            "salience": round(m.salience, 2),
            "confidence": round(m.confidence, 2),
            "usage": m.usage_count,
            "contradiction": round(m.contradiction_score, 2),
        }
        for m in memory.list_memories()
    ]
    st.dataframe(rows, use_container_width=True) if rows else st.info("No memories yet.")


def render_quality(memory: MemoryOS) -> None:
    st.header("Memory Quality")
    rows = [
        {
            "id": r.memory_id,
            "utility": round(r.utility_score, 3),
            "recommendation": r.recommendation,
            "reasons": "; ".join(r.reasons),
        }
        for r in memory.evaluate_all()
    ]
    st.dataframe(rows, use_container_width=True) if rows else st.info("No memories yet.")


def render_sleep(memory: MemoryOS) -> None:
    st.header("Sleep Reports")
    if st.button("Run a sleep cycle now"):
        report = memory.sleep()
        st.success(report.summary)
        st.json(report.model_dump())
    else:
        st.caption("Press the button to consolidate memory and generate a report.")


def render_failures(memory: MemoryOS) -> None:
    st.header("Failure Diagnostics")
    error_text = st.text_input(
        "Diagnose an error message", "ModuleNotFoundError: No module named 'foo'"
    )
    if error_text:
        plan = heal(error_text)
        st.json(plan.model_dump())
    st.subheader("Failures found in traces")
    diagnoses = detect_failures(memory.load_traces())
    rows = [{"type": d.failure_type, "root_cause": d.root_cause} for d in diagnoses]
    st.dataframe(rows, use_container_width=True) if rows else st.info("No failures in traces.")


def render_market(memory: MemoryOS) -> None:  # noqa: ARG001 - uniform page signature
    st.header("Reasoning Market")
    task = st.text_input("Task", "Choose the best repair strategy for a schema error")
    if task:
        result = ReasoningMarket().deliberate(task)
        st.metric("Winner", result.winner.critic_name)
        st.write(result.winner.action)
        st.dataframe(
            [
                {
                    "critic": p.critic_name,
                    "truth": p.truth_value,
                    "reward": p.expected_reward,
                    "risk": p.risk,
                    "cost": p.cost,
                }
                for p in result.ranked
            ],
            use_container_width=True,
        )
        st.code("\n".join(result.audit_trail))


def render_skills(memory: MemoryOS) -> None:
    st.header("Skills")
    rows = [
        {
            "name": s.name,
            "description": s.description,
            "usage": s.usage_count,
            "confidence": round(s.confidence, 2),
        }
        for s in memory.skills.list_skills()
    ]
    st.dataframe(rows, use_container_width=True) if rows else st.info("No skills yet.")


def render_energy(memory: MemoryOS) -> None:
    st.header("Energy Report")
    energy = build_energy_report(memory.list_memories(), memory.load_traces())
    st.json(energy.model_dump())


def main() -> None:
    st.set_page_config(page_title="Plasticity Agent Runtime", layout="wide")
    st.title("🧠 Plasticity Agent Runtime")
    st.caption(f"Memory directory: {MEMORY_DIR}")

    pages = {
        "Overview": render_overview,
        "Memories": render_memories,
        "Memory Quality": render_quality,
        "Sleep Reports": render_sleep,
        "Failure Diagnostics": render_failures,
        "Reasoning Market": render_market,
        "Skills": render_skills,
        "Energy Report": render_energy,
    }
    choice = st.sidebar.radio("Page", list(pages.keys()))

    memory = _open()
    try:
        pages[choice](memory)
    finally:
        memory.close()


main()
