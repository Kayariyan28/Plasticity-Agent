"""Plasticity Agent Runtime.

Neuroplastic memory, self-healing, and critical reasoning for AI agents.

A local-first, framework-agnostic runtime that gives agents an evolving memory,
sleep-like consolidation, reflection, advisory self-healing, a critic-driven
reasoning market, a skill library, and thermodynamic-style reliability reporting.

Scientific framings (complementary learning systems, synaptic homeostasis,
Reflexion, Self-Refine, Voyager-style skills, auction theory, the free-energy
principle) are used as *software metaphors and design principles* — not claims
of biological equivalence or consciousness.
"""

from __future__ import annotations

__version__ = "0.2.0"

from plasticity_agent.core.agent import PlasticAgent
from plasticity_agent.core.config import PlasticityConfig
from plasticity_agent.core.events import TraceEvent
from plasticity_agent.core.runtime import RunResult, Runtime
from plasticity_agent.core.trace import Tracer
from plasticity_agent.healing.detector import FailureDetector, detect_failures
from plasticity_agent.healing.diagnosis import FailureDiagnosis, diagnose
from plasticity_agent.healing.repair import RepairPlan, heal, plan_repair
from plasticity_agent.healing.sandbox import RepairConsent, Sandbox, SandboxResult
from plasticity_agent.learning.curriculum import Curriculum, CurriculumItem
from plasticity_agent.learning.reward import normalize_reward, shape_reward
from plasticity_agent.learning.skill_library import Skill, SkillLibrary
from plasticity_agent.llm.client import LLMClient, coerce_llm
from plasticity_agent.memory.consolidation import SleepReport
from plasticity_agent.memory.contradiction import ContradictionChecker, detect_contradiction
from plasticity_agent.memory.embeddings import (
    EmbeddingBackend,
    HashingEmbeddingBackend,
    SentenceTransformerBackend,
    get_embedder,
)
from plasticity_agent.memory.memory_os import (
    MemoryOS,
    compute_utility_score,
    score_memory_quality,
)
from plasticity_agent.memory.salience import calculate_salience
from plasticity_agent.memory.schemas import (
    Memory,
    MemoryQualityReport,
    MemorySearchResult,
)
from plasticity_agent.memory.vector_index import VectorIndex
from plasticity_agent.metrics.tracker import (
    ImprovementReport,
    ImprovementTracker,
    MetricSnapshot,
)
from plasticity_agent.observability.otel import OTelExporter
from plasticity_agent.reasoning.auction import AuctionResult, run_auction, score_proposal
from plasticity_agent.reasoning.critic import Critic, Proposal
from plasticity_agent.reasoning.critics import LLMCritic
from plasticity_agent.reasoning.debate import Debate, DebateResult
from plasticity_agent.reasoning.market import ReasoningMarket
from plasticity_agent.reflection.lessons import Lesson, ReflectionInput
from plasticity_agent.reflection.reflector import Reflector
from plasticity_agent.reflection.self_refine import SelfRefine, SelfRefineResult
from plasticity_agent.thermodynamics.energy_report import EnergyReport, build_energy_report

__all__ = [
    "__version__",
    # core
    "PlasticAgent",
    "PlasticityConfig",
    "Runtime",
    "RunResult",
    "Tracer",
    "TraceEvent",
    # memory
    "MemoryOS",
    "Memory",
    "MemoryQualityReport",
    "MemorySearchResult",
    "SleepReport",
    "calculate_salience",
    "detect_contradiction",
    "ContradictionChecker",
    "compute_utility_score",
    "score_memory_quality",
    # retrieval / embeddings
    "EmbeddingBackend",
    "HashingEmbeddingBackend",
    "SentenceTransformerBackend",
    "get_embedder",
    "VectorIndex",
    # llm
    "LLMClient",
    "coerce_llm",
    # reflection
    "Reflector",
    "Lesson",
    "ReflectionInput",
    "SelfRefine",
    "SelfRefineResult",
    # healing
    "FailureDiagnosis",
    "diagnose",
    "RepairPlan",
    "plan_repair",
    "heal",
    "Sandbox",
    "SandboxResult",
    "RepairConsent",
    "FailureDetector",
    "detect_failures",
    # reasoning
    "ReasoningMarket",
    "Proposal",
    "Critic",
    "LLMCritic",
    "AuctionResult",
    "run_auction",
    "score_proposal",
    "Debate",
    "DebateResult",
    # learning
    "Skill",
    "SkillLibrary",
    "shape_reward",
    "normalize_reward",
    "Curriculum",
    "CurriculumItem",
    # thermodynamics
    "EnergyReport",
    "build_energy_report",
    # metrics & observability
    "ImprovementTracker",
    "MetricSnapshot",
    "ImprovementReport",
    "OTelExporter",
]
