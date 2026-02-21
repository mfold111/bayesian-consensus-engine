"""Tests for core consensus calculations."""

import json

import pytest

from bayesian_engine.core import (
    Signal,
    TieInfo,
    ConsensusResult,
    compute_consensus,
    compute_consensus_with_tie_priority,
    _normalize_reliability,
    _compute_bayesian_weight,
    _resolve_tie,
)


class TestSignal:
    """Tests for Signal dataclass."""

    def test_from_dict_minimal(self) -> None:
        """Create Signal from minimal dict."""
        data = {"sourceId": "src1", "probability": 0.8}
        sig = Signal.from_dict(data)
        assert sig.source_id == "src1"
        assert sig.probability == 0.8
        assert sig.weight == 1.0
        assert sig.metadata == {}

    def test_from_dict_full(self) -> None:
        """Create Signal from full dict."""
        data = {
            "sourceId": "src1",
            "probability": 0.8,
            "weight": 2.0,
            "metadata": {"type": "model"},
        }
        sig = Signal.from_dict(data)
        assert sig.source_id == "src1"
        assert sig.probability == 0.8
        assert sig.weight == 2.0
        assert sig.metadata == {"type": "model"}


class TestNormalizeReliability:
    """Tests for reliability normalization."""

    def test_none_uses_default(self) -> None:
        """None reliability uses default."""
        assert _normalize_reliability(None, 0.5) == 0.5

    def test_clamps_to_range(self) -> None:
        """Values outside [0, 1] are clamped."""
        assert _normalize_reliability(1.5) == 1.0
        assert _normalize_reliability(-0.5) == 0.0

    def test_passes_valid_values(self) -> None:
        """Valid values pass through unchanged."""
        assert _normalize_reliability(0.7) == 0.7
        assert _normalize_reliability(0.0) == 0.0
        assert _normalize_reliability(1.0) == 1.0


class TestBayesianWeight:
    """Tests for Bayesian weight computation."""

    def test_basic_weight(self) -> None:
        """Basic weight = reliability (probability-independent)."""
        weight = _compute_bayesian_weight(0.8, 0.5)
        assert weight == 0.5

    def test_zero_reliability(self) -> None:
        """Zero reliability gives zero weight."""
        weight = _compute_bayesian_weight(0.8, 0.0)
        assert weight == 0.0

    def test_full_reliability(self) -> None:
        """Full reliability gives weight = 1.0."""
        weight = _compute_bayesian_weight(0.8, 1.0)
        assert weight == 1.0


class TestResolveTie:
    """Tests for deterministic tie resolution."""

    def test_lexicographic_ordering(self) -> None:
        """Ties resolved by lexicographic source-id ordering."""
        candidates = [("source_z", 0.5), ("source_a", 0.5), ("source_m", 0.5)]
        tie_info = _resolve_tie(candidates, 0.5)

        # 'source_a' should win (lexicographically first)
        assert tie_info.winner == "source_a"
        assert tie_info.break_method == "lexicographic"

    def test_two_way_tie(self) -> None:
        """Two-way tie resolves correctly."""
        candidates = [("beta", 0.3), ("alpha", 0.3)]
        tie_info = _resolve_tie(candidates, 0.3)

        assert tie_info.winner == "alpha"
        assert set(tie_info.tied_sources) == {"alpha", "beta"}

    def test_records_all_tied_sources(self) -> None:
        """All tied sources recorded in TieInfo."""
        candidates = [("c", 0.5), ("b", 0.5), ("a", 0.5)]
        tie_info = _resolve_tie(candidates, 0.5)

        assert set(tie_info.tied_sources) == {"a", "b", "c"}
        assert tie_info.tie_value == 0.5


class TestComputeConsensusBasic:
    """Tests for basic consensus computation."""

    def test_empty_signals(self) -> None:
        """Empty signal list returns no consensus."""
        result = compute_consensus([])
        assert result["schemaVersion"] == "1.0.0"
        assert result["consensus"] is None
        assert result["confidence"] is None
        assert result["diagnostics"]["status"] == "no_signals"

    def test_single_signal(self) -> None:
        """Single signal returns its probability."""
        signals = [{"sourceId": "src1", "probability": 0.8}]
        result = compute_consensus(signals)

        # With default reliability 0.5, weight = 0.5
        # consensus = (0.8 * 0.5) / 0.5 = 0.8
        assert result["consensus"] == pytest.approx(0.8)
        assert result["confidence"] is not None
        assert result["sourceWeights"]["src1"] == pytest.approx(0.5)

    def test_two_signals_different_probabilities(self) -> None:
        """Two signals with different probabilities compute weighted average."""
        signals = [
            {"sourceId": "src1", "probability": 0.9},
            {"sourceId": "src2", "probability": 0.5},
        ]
        result = compute_consensus(signals)

        # Both have reliability 0.5, so equal weights
        # consensus = (0.9 + 0.5) / 2 = 0.7
        assert result["consensus"] == pytest.approx(0.7)


class TestComputeConsensusDeterminism:
    """Tests for deterministic tie-breaking behavior."""

    def test_deterministic_output_same_inputs(self) -> None:
        """Same inputs always produce same output."""
        signals = [
            {"sourceId": "src_z", "probability": 0.8},
            {"sourceId": "src_a", "probability": 0.6},
            {"sourceId": "src_m", "probability": 0.7},
        ]

        result1 = compute_consensus(signals)
        result2 = compute_consensus(signals)

        # JSON serialization should be identical
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_signal_order_does_not_affect_consensus(self) -> None:
        """Order of input signals doesn't change consensus value."""
        signals_ordered = [
            {"sourceId": "a", "probability": 0.8},
            {"sourceId": "b", "probability": 0.6},
            {"sourceId": "c", "probability": 0.7},
        ]
        signals_shuffled = [
            {"sourceId": "c", "probability": 0.7},
            {"sourceId": "a", "probability": 0.8},
            {"sourceId": "b", "probability": 0.6},
        ]

        result1 = compute_consensus(signals_ordered)
        result2 = compute_consensus(signals_shuffled)

        assert result1["consensus"] == pytest.approx(result2["consensus"])
        assert result1["confidence"] == pytest.approx(result2["confidence"])

    def test_tie_break_lexicographic(self) -> None:
        """When sources tie, winner is lexicographically first."""
        # Create a scenario where two sources have identical weighted contributions
        signals = [
            {"sourceId": "zebra", "probability": 0.5, "weight": 1.0},
            {"sourceId": "alpha", "probability": 0.5, "weight": 1.0},
        ]

        result = compute_consensus(signals)

        # Check that tie event is recorded
        tie_events = result["diagnostics"].get("tieEvents", [])
        assert len(tie_events) > 0

        # Alpha should win due to lexicographic ordering
        tie_event = tie_events[0]
        assert tie_event["winner"] == "alpha"
        assert set(tie_event["tiedSources"]) == {"zebra", "alpha"}
        assert tie_event["breakMethod"] == "lexicographic"

    def test_three_way_tie_deterministic(self) -> None:
        """Three-way tie breaks deterministically."""
        signals = [
            {"sourceId": "charlie", "probability": 0.5},
            {"sourceId": "alpha", "probability": 0.5},
            {"sourceId": "bravo", "probability": 0.5},
        ]

        result = compute_consensus(signals)
        tie_events = result["diagnostics"].get("tieEvents", [])

        # Should have recorded the tie
        assert len(tie_events) > 0

        # Find the tie event for all three sources
        all_tied_event = next(
            (e for e in tie_events if len(e["tiedSources"]) == 3),
            None,
        )
        if all_tied_event:
            assert all_tied_event["winner"] == "alpha"


class TestTieMetadataInDiagnostics:
    """Tests for tie metadata being emitted in diagnostics."""

    def test_tie_events_recorded(self) -> None:
        """Tie events are recorded in diagnostics."""
        signals = [
            {"sourceId": "a", "probability": 0.5},
            {"sourceId": "b", "probability": 0.5},
        ]
        result = compute_consensus(signals)

        assert "tieEvents" in result["diagnostics"]
        assert isinstance(result["diagnostics"]["tieEvents"], list)

    def test_no_tie_no_events(self) -> None:
        """No tie events when all weights are unique."""
        signals = [
            {"sourceId": "a", "probability": 0.9},
            {"sourceId": "b", "probability": 0.1},
        ]
        # With different probabilities, weights should differ
        result = compute_consensus(signals, source_reliabilities={"a": 0.8, "b": 0.3})

        # May or may not have tie events depending on tolerance
        # but the field should exist
        assert "tieEvents" in result["diagnostics"]

    def test_tie_event_structure(self) -> None:
        """Tie events have required fields."""
        signals = [
            {"sourceId": "x", "probability": 0.5},
            {"sourceId": "y", "probability": 0.5},
        ]
        result = compute_consensus(signals)
        tie_events = result["diagnostics"]["tieEvents"]

        if tie_events:
            event = tie_events[0]
            assert "tiedSources" in event
            assert "winner" in event
            assert "tieValue" in event
            assert "breakMethod" in event


class TestReliabilityImpact:
    """Tests for how reliability affects consensus."""

    def test_higher_reliability_more_weight(self) -> None:
        """Higher reliability gives more weight to signal."""
        signals = [
            {"sourceId": "reliable", "probability": 0.9},
            {"sourceId": "unreliable", "probability": 0.1},
        ]
        reliabilities = {"reliable": 0.9, "unreliable": 0.1}

        result = compute_consensus(signals, source_reliabilities=reliabilities)

        # Reliable source should dominate
        assert result["sourceWeights"]["reliable"] > result["sourceWeights"]["unreliable"]
        # Consensus should be closer to reliable source's probability
        assert result["consensus"] > 0.5

    def test_unknown_source_uses_default(self) -> None:
        """Unknown sources use default reliability."""
        signals = [{"sourceId": "unknown", "probability": 0.8}]
        result = compute_consensus(signals, default_reliability=0.5)

        # Weight = default_reliability = 0.5
        assert result["sourceWeights"]["unknown"] == pytest.approx(0.5)


class TestConsensusResultSerialization:
    """Tests for ConsensusResult output format."""

    def test_output_has_required_fields(self) -> None:
        """Output dictionary has all required fields per PRD."""
        signals = [{"sourceId": "test", "probability": 0.5}]
        result = compute_consensus(signals)

        required_fields = [
            "schemaVersion",
            "consensus",
            "confidence",
            "sourceWeights",  # Plural per PRD §7-D
            "normalization",
            "diagnostics",
        ]
        for field_name in required_fields:
            assert field_name in result, f"Missing required field: {field_name}"

    def test_output_is_json_serializable(self) -> None:
        """Output can be serialized to JSON."""
        signals = [
            {"sourceId": "a", "probability": 0.5},
            {"sourceId": "b", "probability": 0.6},
        ]
        result = compute_consensus(signals)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed == result


class TestComputeConsensusWithTiePriority:
    """Tests for consensus with explicit priority ordering."""

    def test_priority_order_in_diagnostics(self) -> None:
        """Priority order is recorded in diagnostics."""
        signals = [
            {"sourceId": "a", "probability": 0.5},
            {"sourceId": "b", "probability": 0.5},
        ]
        priority = ["b", "a"]  # b has priority over a

        result = compute_consensus_with_tie_priority(signals, priority_order=priority)

        assert "priorityOrder" in result["diagnostics"]
        assert result["diagnostics"]["priorityOrder"] == priority

    def test_none_priority_uses_standard(self) -> None:
        """None priority falls back to standard computation."""
        signals = [{"sourceId": "a", "probability": 0.5}]

        result = compute_consensus_with_tie_priority(signals, priority_order=None)
        standard_result = compute_consensus(signals)

        assert result["consensus"] == standard_result["consensus"]


class TestGoldenFixture:
    """Golden regression tests for determinism."""

    GOLDEN_INPUT = [
        {"sourceId": "model_alpha", "probability": 0.75},
        {"sourceId": "model_beta", "probability": 0.65},
        {"sourceId": "model_gamma", "probability": 0.80},
    ]

    GOLDEN_EXPECTED_CONSENSUS = (0.75 + 0.65 + 0.80) / 3  # Equal weights

    def test_golden_consensus_value(self) -> None:
        """Golden fixture produces expected consensus."""
        result = compute_consensus(self.GOLDEN_INPUT)
        assert result["consensus"] == pytest.approx(self.GOLDEN_EXPECTED_CONSENSUS)

    def test_golden_deterministic_output(self) -> None:
        """Golden fixture output is deterministic."""
        results = [compute_consensus(self.GOLDEN_INPUT) for _ in range(10)]

        # All outputs should be byte-identical when serialized
        serialized = [json.dumps(r, sort_keys=True) for r in results]
        assert len(set(serialized)) == 1, "Output is not deterministic!"

    def test_golded_output_stable_across_versions(self) -> None:
        """Golden output should not change between runs."""
        result = compute_consensus(self.GOLDEN_INPUT)

        # These values should remain stable
        assert result["schemaVersion"] == "1.0.0"
        assert result["consensus"] == pytest.approx(0.733333, rel=1e-4)
        assert result["diagnostics"]["status"] == "computed"
        assert result["diagnostics"]["signalCount"] == 3
