"""Tests for config module constants."""

import pytest

from bayesian_engine.config import (
    DEFAULT_RELIABILITY,
    DEFAULT_CONFIDENCE,
    MAX_UPDATE_STEP,
    TIE_TOLERANCE,
    DECAY_RATE,
    DECAY_INTERVAL_DAYS,
)


class TestConfigDefaults:
    """Tests for cold-start default constants per PRD §9."""

    def test_default_reliability_value(self) -> None:
        """Default reliability is 0.50 per PRD."""
        assert DEFAULT_RELIABILITY == 0.50

    def test_default_confidence_value(self) -> None:
        """Default confidence is 0.25 per PRD."""
        assert DEFAULT_CONFIDENCE == 0.25

    def test_max_update_step_value(self) -> None:
        """Max update step is 0.10 per PRD."""
        assert MAX_UPDATE_STEP == 0.10

    def test_defaults_in_valid_range(self) -> None:
        """All defaults are in valid [0, 1] range."""
        assert 0.0 <= DEFAULT_RELIABILITY <= 1.0
        assert 0.0 <= DEFAULT_CONFIDENCE <= 1.0
        assert 0.0 <= MAX_UPDATE_STEP <= 1.0


class TestTieTolerance:
    """Tests for tie detection tolerance."""

    def test_tolerance_is_small(self) -> None:
        """Tolerance should be very small for precision."""
        assert TIE_TOLERANCE < 1e-6
        assert TIE_TOLERANCE > 0

    def test_tolerance_detects_near_equal(self) -> None:
        """Values within tolerance should be considered equal."""
        val1 = 0.5
        val2 = 0.5 + TIE_TOLERANCE / 2
        assert abs(val1 - val2) < TIE_TOLERANCE


class TestDecayConfig:
    """Tests for reliability decay configuration."""

    def test_decay_rate_positive(self) -> None:
        """Decay rate should be positive."""
        assert DECAY_RATE > 0

    def test_decay_interval_positive(self) -> None:
        """Decay interval should be positive."""
        assert DECAY_INTERVAL_DAYS > 0

    def test_decay_rate_is_small(self) -> None:
        """Decay rate should be gradual."""
        assert DECAY_RATE < 0.1
