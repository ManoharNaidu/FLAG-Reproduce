"""Text-augmented fraud datasets (YelpChi, Amazon) in one schema.

IMPORTANT: these are NOT part of the FLAG reproduction. The FLAG paper reports
no YelpChi/Amazon numbers and states both datasets lack textual information.
Everything here is a novel, separately-labelled experiment -- see
research/evidence_matrix.md section 7.
"""
from flagbench.fraud_text.base import (
    CANONICAL, DERIVED, UNRESOLVED, VERIFIED,
    DatasetIntegrityError, FLAGDataset,
)
from flagbench.fraud_text.loaders import load_amazon, load_dataset, load_yelpchi
from flagbench.fraud_text.serialization import load, save
from flagbench.fraud_text.validation import ValidationReport, validate_dataset

__all__ = [
    "FLAGDataset", "DatasetIntegrityError",
    "CANONICAL", "DERIVED", "VERIFIED", "UNRESOLVED",
    "load_dataset", "load_yelpchi", "load_amazon",
    "save", "load", "validate_dataset", "ValidationReport",
]
