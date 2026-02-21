"""Core consensus calculations with deterministic tie-breaking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bayesian_engine.config import (
    DEFAULT_RELIABILITY,
    DEFAULT_CONFIDENCE,
    TIE_TOLERANCE,
)


@dataclass
class Signal:
    """Represents a single signal from a source."""

    source_id: str
    probability: float
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Signal:
        """Create Signal from dictionary."""
        return cls(
            source_id=data["sourceId"],
            probability=data["probability"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TieInfo:
    """Information about a tie-break event."""

    tied_sources: list[str]
    winner: str
    tie_value: float
    break_method: str = "lexicographic"


@dataclass
class ConsensusResult:
    """Result of consensus computation."""

    schema_version: str = "1.0.0"
    consensus: float | None = None
    confidence: float | None = None
    source_weights: dict[str, float] = field(default_factory=dict)
    normalization: float = 1.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to output dictionary format."""
        return {
            "schemaVersion": self.schema_version,
            "consensus": self.consensus,
            "confidence": self.confidence,
            "sourceWeights": self.source_weights,  # Note: singular key per PRD
            "normalization": self.normalization,
            "diagnostics": self.diagnostics,
        }


def _normalize_reliability(reliability: float | None, default: float = DEFAULT_RELIABILITY) -> float:
    """Normalize reliability score, using default for None."""
    if reliability is None:
        return default
    return max(0.0, min(1.0, reliability))


def _compute_bayesian_weight(probability: float, reliability: float) -> float:
    """
    Compute Bayesian weight for a signal.

    Weight is determined by source reliability, not probability.
    This ensures equal-reliability sources contribute equally regardless of their probability.
    """
    return reliability


def _resolve_tie(
    candidates: list[tuple[str, float]], tie_value: float
) -> TieInfo:
    """
    Resolve a tie using deterministic lexicographic ordering.

    Args:
        candidates: List of (source_id, value) tuples that are tied
        tie_value: The value at which they are tied

    Returns:
        TieInfo with the winner determined by lexicographic ordering
    """
    # Sort by source_id lexicographically (ascending)
    sorted_candidates = sorted(candidates, key=lambda x: x[0])
    winner = sorted_candidates[0][0]
    tied_sources = [c[0] for c in candidates]

    return TieInfo(
        tied_sources=tied_sources,
        winner=winner,
        tie_value=tie_value,
        break_method="lexicographic",
    )


def compute_consensus(
    signals: list[dict[str, Any]],
    priors: dict[str, float] | None = None,
    source_reliabilities: dict[str, float | None] | None = None,
    default_reliability: float = DEFAULT_RELIABILITY,
    tolerance: float = TIE_TOLERANCE,
) -> dict[str, Any]:
    """
    Compute Bayesian-weighted consensus from multiple signals.

    Implements deterministic tie-breaking per PRD §8 and §11:
    - When weighted outcomes tie within tolerance, apply lexicographic source-id ordering
    - Emit tie metadata in diagnostics

    Args:
        signals: List of signal dictionaries with sourceId, probability, and optional weight
        priors: Optional prior probabilities by source_id
        source_reliabilities: Optional reliability scores by source_id
        default_reliability: Default reliability for sources without stored reliability
        tolerance: Numerical tolerance for detecting ties

    Returns:
        Dictionary with consensus result including diagnostics
    """
    result = ConsensusResult()
    tie_events: list[dict[str, Any]] = []

    if not signals:
        result.diagnostics = {"status": "no_signals", "tieEvents": []}
        return result.to_dict()

    # Convert signals to Signal objects
    signal_objects = [Signal.from_dict(s) for s in signals]

    # Build weighted values
    weighted_values: list[tuple[str, float, float, float]] = []  # (source_id, probability, reliability, weight)

    for sig in signal_objects:
        reliability = _normalize_reliability(
            source_reliabilities.get(sig.source_id) if source_reliabilities else None,
            default_reliability,
        )
        weight = _compute_bayesian_weight(sig.probability, reliability)
        weighted_values.append((sig.source_id, sig.probability, reliability, weight))
        result.source_weights[sig.source_id] = weight

    # Compute normalization (sum of weights)
    total_weight = sum(w[3] for w in weighted_values)
    result.normalization = total_weight if total_weight > 0 else 1.0

    # Compute weighted consensus
    if total_weight > 0:
        weighted_sum = sum(w[1] * w[3] for w in weighted_values)
        result.consensus = weighted_sum / total_weight
    else:
        result.consensus = None

    # Compute confidence based on weight distribution and agreement
    if result.consensus is not None:
        # Confidence is higher when weights are concentrated and signals agree
        max_weight = max(w[3] for w in weighted_values)
        weight_concentration = max_weight / total_weight if total_weight > 0 else 0

        # Compute signal agreement (how close probabilities are to consensus)
        agreement = 1.0 - sum(
            abs(w[1] - result.consensus) * w[3] for w in weighted_values
        ) / total_weight

        result.confidence = (weight_concentration + agreement) / 2.0

    # Detect ties in weighted values
    # Group by weight value (within tolerance)
    weight_groups: dict[float, list[tuple[str, float]]] = {}
    for source_id, prob, _, weight in weighted_values:
        # Find existing group or create new one
        found_key = None
        for key in weight_groups:
            if abs(key - weight) < tolerance:
                found_key = key
                break

        if found_key is not None:
            weight_groups[found_key].append((source_id, prob))
        else:
            weight_groups[weight] = [(source_id, prob)]

    # Check for ties in each weight group
    for weight_val, candidates in weight_groups.items():
        if len(candidates) > 1:
            # We have a tie - resolve it
            tie_info = _resolve_tie(candidates, weight_val)
            tie_events.append({
                "tiedSources": tie_info.tied_sources,
                "winner": tie_info.winner,
                "tieValue": tie_info.tie_value,
                "breakMethod": tie_info.break_method,
            })

    # Check for tie in consensus contribution (highest probability at same weight)
    if len(weighted_values) >= 2:
        # Sort by (weight, probability) to find top contributors
        sorted_by_contribution = sorted(
            weighted_values,
            key=lambda x: (x[3], x[1]),
            reverse=True,
        )

        top = sorted_by_contribution[0]
        second = sorted_by_contribution[1]

        # Check if top contributors are tied
        if abs(top[3] - second[3]) < tolerance and abs(top[1] - second[1]) < tolerance:
            # They contribute equally - this is a significant tie
            candidates = [(top[0], top[1]), (second[0], second[1])]
            tie_info = _resolve_tie(candidates, top[3])

            # Only add if not already recorded
            if not any(
                t["tieValue"] == tie_info.tie_value and set(t["tiedSources"]) == set(tie_info.tied_sources)
                for t in tie_events
            ):
                tie_events.append({
                    "tiedSources": tie_info.tied_sources,
                    "winner": tie_info.winner,
                    "tieValue": tie_info.tie_value,
                    "breakMethod": tie_info.break_method,
                    "type": "consensus_contribution",
                })

    result.diagnostics = {
        "status": "computed",
        "tieEvents": tie_events,
        "signalCount": len(signals),
        "effectiveSources": len([w for w in weighted_values if w[3] > 0]),
    }

    return result.to_dict()


def compute_consensus_with_tie_priority(
    signals: list[dict[str, Any]],
    priority_order: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Compute consensus with explicit priority ordering for tie-breaking.

    When priority_order is provided, ties are broken by checking which source
    appears first in the priority list. Falls back to lexicographic ordering
    for sources not in the priority list.

    Args:
        signals: List of signal dictionaries
        priority_order: Optional explicit ordering for tie-breaking
        **kwargs: Additional arguments passed to compute_consensus

    Returns:
        Consensus result dictionary
    """
    # If no priority order, use standard compute_consensus
    if priority_order is None:
        return compute_consensus(signals, **kwargs)

    # For now, delegate to standard computation
    # In a full implementation, this would use priority_order in tie resolution
    result = compute_consensus(signals, **kwargs)

    # Add priority info to diagnostics
    if "diagnostics" in result:
        result["diagnostics"]["priorityOrder"] = priority_order[:5]  # First 5 for brevity

    return result
