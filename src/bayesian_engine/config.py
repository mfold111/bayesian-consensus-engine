"""Configuration constants for Bayesian Engine.

This module centralizes deterministic constants for reliability cold-start
and update behavior per PRD §9.

All constants are documented and should not change without a version bump.
"""

# Cold-start defaults for unseen sources
DEFAULT_RELIABILITY = 0.50
"""float: Default reliability score for sources without stored reliability.

Per PRD §9, this is the initial trust level assigned to new sources.
Value of 0.50 represents neutral/unknown reliability.
"""

DEFAULT_CONFIDENCE = 0.25
"""float: Default confidence for consensus results with minimal data.

Per PRD §9, this represents low confidence when insufficient evidence exists.
Value of 0.25 indicates below-neutral confidence.
"""

# Reliability update constraints
MAX_UPDATE_STEP = 0.10
"""float: Maximum single-step change to reliability score.

Per PRD §9, this caps how much reliability can change in a single update,
preventing wild swings from single outcomes.
Value of 0.10 means reliability can change by at most ±10% per update.
"""

# Numerical tolerances
TIE_TOLERANCE = 1e-9
"""float: Numerical tolerance for detecting ties in weighted values.

Per PRD §8, values within this tolerance are considered equal for tie-breaking.
"""

# Decay configuration (for future reliability decay implementation)
DECAY_RATE = 0.01
"""float: Exponential decay rate for reliability over time.

Per PRD §7B, reliability decays exponentially. This rate determines how quickly.
Value of 0.01 means ~1% decay per time unit.
"""

DECAY_INTERVAL_DAYS = 7
"""int: Time interval (in days) between decay applications.

Per PRD §7B, decay is applied periodically, not continuously.
"""
