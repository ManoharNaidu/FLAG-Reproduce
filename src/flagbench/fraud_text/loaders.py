"""One API for both datasets (brief sections 17, 38, 42).

    from flagbench.fraud_text import load_dataset
    ds = load_dataset("yelpchi")    # node_type == "review"
    ds = load_dataset("amazon")     # node_type == "user"

Both return the same `FLAGDataset`, so downstream code never branches on the
dataset. Loading prefers the serialized payload and falls back to building from
source, so an expensive build is never repeated (brief section 45).
"""
from __future__ import annotations

from flagbench.fraud_text import amazon as _amazon
from flagbench.fraud_text import serialization as _ser
from flagbench.fraud_text import yelpchi as _yelpchi
from flagbench.fraud_text.base import FLAGDataset

BUILDERS = {"yelpchi": _yelpchi.build, "amazon": _amazon.build}


def load_dataset(name: str, rebuild: bool = False) -> FLAGDataset:
    if name not in BUILDERS:
        raise KeyError(f"unknown dataset {name!r}; expected one of {sorted(BUILDERS)}")
    if not rebuild:
        try:
            return _ser.load(name)
        except FileNotFoundError:
            pass
    return BUILDERS[name]()


def load_yelpchi(rebuild: bool = False) -> FLAGDataset:
    return load_dataset("yelpchi", rebuild=rebuild)


def load_amazon(rebuild: bool = False) -> FLAGDataset:
    return load_dataset("amazon", rebuild=rebuild)
