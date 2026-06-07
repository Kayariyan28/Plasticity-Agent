"""Server: FastAPI app + Streamlit dashboard entry points.

Only :mod:`api` is imported here; the Streamlit ``dashboard`` module is a
standalone script (importing it requires ``streamlit``) and is launched by the
CLI via ``streamlit run``.
"""

from __future__ import annotations

from plasticity_agent.server.api import build_app, serve

__all__ = ["build_app", "serve"]
