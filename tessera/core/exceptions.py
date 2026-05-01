"""Structured exception hierarchy for Tessera.

Import from here rather than catching bare Exception so callers can handle
specific failure modes without catching unrelated errors.
"""
from __future__ import annotations


class TesseraError(Exception):
    """Base exception for all Tessera errors."""


class ConfigurationError(TesseraError):
    """Invalid configuration — wrong spec type, n_examples <= 0, etc."""


class GenerationError(TesseraError):
    """LLM generation failed for a single example node."""


class CritiqueError(TesseraError):
    """Critique scoring failed for a single example."""


class TaxonomyError(TesseraError):
    """Taxonomy expansion returned insufficient or unparseable nodes."""


class DeduplicationError(TesseraError):
    """Deduplication step encountered an unrecoverable error."""
