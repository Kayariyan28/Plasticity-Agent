"""Advisory self-healing: detection, diagnosis, repair planning, sandbox."""

from __future__ import annotations

from plasticity_agent.healing.detector import FailureDetector, detect_failures
from plasticity_agent.healing.diagnosis import FailureDiagnosis, FailureType, diagnose
from plasticity_agent.healing.repair import RepairPlan, heal, plan_repair
from plasticity_agent.healing.sandbox import RepairConsent, Sandbox, SandboxResult

__all__ = [
    "FailureDetector",
    "detect_failures",
    "FailureDiagnosis",
    "FailureType",
    "diagnose",
    "RepairPlan",
    "heal",
    "plan_repair",
    "Sandbox",
    "SandboxResult",
    "RepairConsent",
]
