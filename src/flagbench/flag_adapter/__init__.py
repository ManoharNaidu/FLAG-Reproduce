"""Adapters from the unified FLAGDataset to FLAG's existing input contract.

No FLAG core file is modified anywhere in this package (brief section 49).
"""
from flagbench.flag_adapter.dataset_adapter import (
    REGISTRY_KEY, to_flag_payload, write_flag_payload,
)
from flagbench.flag_adapter.graph_adapter import homogeneous_edge_index
from flagbench.flag_adapter.text_adapter import texts_for_flag

__all__ = ["to_flag_payload", "write_flag_payload", "REGISTRY_KEY",
           "homogeneous_edge_index", "texts_for_flag"]
