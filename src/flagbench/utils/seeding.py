"""Seeding and run identity.

The FLAG paper evaluates public datasets **25 times with 5 random seeds and 5
random initializations** (the released code does 5 x 1). Reproducing that
protocol needs the two axes to be genuinely independent, which is the reason
this module exists rather than a bare `torch.manual_seed` call.

  * ``seed``  -- the *data* axis: split construction, benchmark downsampling,
                 subgraph sampling, batch order.
  * ``init``  -- the *parameter* axis: weight initialisation only.

Mixing them (e.g. `manual_seed(seed)` before building the model) means run
(seed=0, init=1) and (seed=1, init=0) are not independent samples, and the
reported standard deviation understates true variance. `derive_seed` keeps the
axes separated by hashing both into a stream-specific value.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 2**32 - 1: numpy's Generator seed ceiling, and safe for torch.
_MAX_SEED = 0xFFFF_FFFF


def derive_seed(*parts: object) -> int:
    """Deterministically derive a 32-bit seed from any hashable description.

    Uses BLAKE2b over the string form rather than Python's ``hash()``, which is
    randomised per process for str/bytes and would break reproducibility across
    runs.

    >>> derive_seed("split", 0) == derive_seed("split", 0)
    True
    >>> derive_seed("split", 0) == derive_seed("split", 1)
    False
    """
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (_MAX_SEED + 1)


@dataclass(frozen=True)
class RunIdentity:
    """The (seed, init) coordinate of one run in the 5x5 grid."""

    seed: int
    init: int = 0
    dataset: str = ""
    model: str = ""
    variant: str = ""
    extra: dict = field(default_factory=dict)

    def stream(self, name: str) -> int:
        """Seed for an independent random stream within this run.

        ``name`` identifies the consumer, e.g. "split", "downsample", "sampling",
        "batch_order", "init". Different names give uncorrelated streams; the
        same name always gives the same value.
        """
        if name == "init":
            # The init axis must NOT depend on `seed`, or the two axes of the
            # 5x5 grid stop being independent.
            return derive_seed("init", self.init, self.model, self.variant)
        return derive_seed(name, self.seed, self.dataset, self.variant)

    def as_record(self) -> dict:
        return {
            "seed": self.seed,
            "initialization": self.init,
            **({"seed_extra": dict(self.extra)} if self.extra else {}),
        }


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy and torch. Returns the seed, for logging.

    ``deterministic`` also requests deterministic cuDNN kernels. That costs
    throughput on GPU but keeps a run reproducible, which matters more here.
    """
    seed = int(seed) % (_MAX_SEED + 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic reductions in some CUDA kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed


def seed_model_init(model_fn, identity: RunIdentity):
    """Build a model under the *init* seed only.

    Keeps parameter initialisation on its own axis, independent of the data seed.

        model = seed_model_init(lambda: GCN(384, 32, 2), identity)
    """
    generator_seed = identity.stream("init")
    with fork_rng(generator_seed):
        return model_fn()


@contextmanager
def fork_rng(seed: int):
    """Temporarily seed all RNGs, restoring the previous state on exit.

    Lets a component (splitting, sampling, init) draw from a reproducible stream
    without perturbing the global sequence the caller relies on.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        seed_everything(seed, deterministic=False)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def numpy_generator(seed: int) -> np.random.Generator:
    """A local numpy Generator. Preferred over the legacy global RandomState."""
    return np.random.default_rng(seed)


def run_grid(
    seeds: int | list[int] = 5,
    inits: int | list[int] = 5,
    **common,
) -> list[RunIdentity]:
    """Build the full (seed x init) grid.

    Defaults reproduce the paper's protocol: 5 seeds x 5 initialisations = 25
    runs. The released code does 5 x 1; see research/decisions.md S-002.

    >>> len(run_grid())
    25
    >>> len(run_grid(seeds=5, inits=1))
    5
    """
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    init_list = list(range(inits)) if isinstance(inits, int) else list(inits)
    return [
        RunIdentity(seed=s, init=i, **common)
        for s in seed_list
        for i in init_list
    ]
