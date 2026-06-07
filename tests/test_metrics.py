"""Tests for cross-run improvement metrics."""

from __future__ import annotations

from plasticity_agent import PlasticAgent
from plasticity_agent.metrics.tracker import ImprovementTracker, MetricSnapshot


def test_report_needs_two_snapshots(tmp_path) -> None:
    tracker = ImprovementTracker(str(tmp_path / "metrics.sqlite"))
    try:
        tracker.record(MetricSnapshot(label="only", plasticity_score=50.0))
        report = tracker.report()
        assert report.snapshots == 1
        assert report.improved is False
    finally:
        tracker.close()


def test_detects_improvement(tmp_path) -> None:
    tracker = ImprovementTracker(str(tmp_path / "metrics.sqlite"))
    try:
        tracker.record(
            MetricSnapshot(plasticity_score=40, avg_utility=0.3, contradiction_pressure=0.5)
        )
        tracker.record(
            MetricSnapshot(plasticity_score=70, avg_utility=0.6, contradiction_pressure=0.2)
        )
        report = tracker.report()
        assert report.snapshots == 2
        assert report.improved is True
        assert report.plasticity_delta == 30.0
        assert report.improvement_score > 0.0
    finally:
        tracker.close()


def test_detects_regression(tmp_path) -> None:
    tracker = ImprovementTracker(str(tmp_path / "metrics.sqlite"))
    try:
        tracker.record(
            MetricSnapshot(plasticity_score=80, avg_utility=0.7, contradiction_pressure=0.1)
        )
        tracker.record(
            MetricSnapshot(plasticity_score=50, avg_utility=0.4, contradiction_pressure=0.5)
        )
        report = tracker.report()
        assert report.improved is False
        assert report.improvement_score < 0.0
    finally:
        tracker.close()


def test_agent_checkpoint_and_improvement(agent: PlasticAgent) -> None:
    agent.remember("a useful, well-cited fact", "semantic", tags=["important"], reward=0.9)
    first = agent.checkpoint("start")
    agent.remember("another reliable fact", "semantic", reward=0.8)
    agent.remember("a third solid fact", "semantic", reward=0.7)
    agent.checkpoint("later")

    report = agent.improvement()
    assert report.snapshots == 2
    assert report.first is not None
    assert report.last is not None
    assert first.memories >= 1
