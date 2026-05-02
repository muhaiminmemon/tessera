"""Runtime configuration constants for Tessera.

All environment-variable reads are centralised here so they appear in exactly
one place and are easy to document, override in tests, or move to a config
file later.
"""
from __future__ import annotations

import os


def max_concurrent() -> int:
    """Maximum number of parallel LLM threads.

    Override by setting the ``TESSERA_MAX_CONCURRENT`` environment variable.
    Defaults to 5.
    """
    return int(os.environ.get("TESSERA_MAX_CONCURRENT", "5"))


# Generation temperatures — kept as named constants so pipeline code never
# contains bare magic numbers.
GENERATION_TEMPERATURE: float = 0.9
CRITIQUE_TEMPERATURE: float = 0.2
TAXONOMY_TEMPERATURE: float = 0.7
CONTEXT_GENERATION_TEMPERATURE: float = 0.9
QA_PAIR_TEMPERATURE: float = 0.7
