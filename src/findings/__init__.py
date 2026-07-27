"""
Findings package: structured extraction and cross-provider consensus.

Provider answers arrive as prose. This package converts them into
comparable structure — recommendations, prices, citations, evidence,
confidence — and merges that structure across providers so agreement is
measured over concrete claims rather than over wording.
"""

from .aggregator import (
    ConsensusAggregate,
    ProductConsensus,
    aggregate_findings,
)
from .extractor import (
    extract_citations,
    extract_findings,
    extract_price,
    score_confidence,
)
from .models import (
    Citation,
    Price,
    ProviderFindings,
    Recommendation,
    normalize_name,
)

__all__ = [
    "Citation",
    "ConsensusAggregate",
    "Price",
    "ProductConsensus",
    "ProviderFindings",
    "Recommendation",
    "aggregate_findings",
    "extract_citations",
    "extract_findings",
    "extract_price",
    "normalize_name",
    "score_confidence",
]
