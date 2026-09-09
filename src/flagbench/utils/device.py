"""Device selection and placement.

Phase 15/16 requirement: CPU support must be real, and CUDA must never be
assumed. Upstream FLAG hardcodes `.cuda()` at ~40 sites and cannot run without a
GPU at all; nothing in this package may do that.

Rules enforced here:
  * `torch.cuda.is_available()` is never assumed True.
  * Asking for a GPU that does not exist is a loud error, never a silent
    fallback to CPU -- a run that quietly changed device would make timing and
    memory numbers meaningless and could mask an OOM-driven config change.
  * The single exception is `device="auto"`, where falling back IS the request.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_CUDA_SPEC = re.compile(r"^cuda(:(?P<index>\d+))?$")


class DeviceUnavailableError(RuntimeError):
    """Raised when a specific device was requested but cannot be provided."""


@dataclass(frozen=True)
class DeviceInfo:
    """What we actually got, for the result record."""

    device: torch.device
    kind: str                  # "cpu" | "cuda"
    name: str                  # human-readable
    index: int | None = None
    total_memory_bytes: int | None = None
    capability: tuple[int, int] | None = None

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def as_record(self) -> dict:
        """Flat dict for the results schema."""
        record = {
            "device": str(self.device),
            "device_kind": self.kind,
            "device_name": self.name,
        }
        if self.total_memory_bytes is not None:
            record["device_memory_gb"] = round(
                self.total_memory_bytes / 1024**3, 2
            )
        if self.capability is not None:
            record["device_capability"] = "%d.%d" % self.capability
        return record

    def __str__(self) -> str:
        if self.total_memory_bytes:
            gb = self.total_memory_bytes / 1024**3
            return f"{self.device} ({self.name}, {gb:.1f} GiB)"
        return f"{self.device} ({self.name})"


def get_device(spec: str | torch.device | None = None) -> torch.device:
    """Resolve a device spec to a `torch.device`.

    Accepts ``"cpu"``, ``"cuda"``, ``"cuda:N"``, ``"auto"``, ``None``, or an
    existing ``torch.device``.

    ``None`` reads ``FLAG_DEVICE`` from the environment, defaulting to ``"cpu"``.
    We default to CPU rather than to ``auto`` so that a run never silently moves
    to a GPU that happens to be present.

    Raises:
        DeviceUnavailableError: a specific CUDA device was asked for and is not
            usable. Never falls back silently.
        ValueError: the spec is not parseable.
    """
    return resolve_device(spec).device


def resolve_device(spec: str | torch.device | None = None) -> DeviceInfo:
    """Like `get_device`, but returns the full `DeviceInfo` record."""
    if spec is None:
        spec = os.environ.get("FLAG_DEVICE", "cpu")
    spec = str(spec).strip().lower()

    if spec == "auto":
        if torch.cuda.is_available():
            return _cuda_info(0)
        logger.info("device=auto and CUDA is unavailable; using CPU")
        return _cpu_info()

    if spec == "cpu":
        return _cpu_info()

    match = _CUDA_SPEC.match(spec)
    if match is None:
        raise ValueError(
            f"unrecognised device spec {spec!r}. "
            f"Expected one of: 'cpu', 'cuda', 'cuda:N', 'auto'."
        )

    if not torch.cuda.is_available():
        raise DeviceUnavailableError(
            f"device {spec!r} was requested but CUDA is not available "
            f"(torch {torch.__version__}). "
            f"This build reports torch.cuda.is_available() == False. "
            f"Use --device cpu, or --device auto to fall back deliberately. "
            f"Refusing to switch device silently: it would invalidate the "
            f"timing and memory fields of this run."
        )

    index = int(match.group("index") or 0)
    count = torch.cuda.device_count()
    if index >= count:
        raise DeviceUnavailableError(
            f"device {spec!r} was requested but only {count} CUDA device(s) "
            f"are visible (valid indices 0..{count - 1}). "
            f"Check CUDA_VISIBLE_DEVICES."
        )
    return _cuda_info(index)


def _cpu_info() -> DeviceInfo:
    return DeviceInfo(
        device=torch.device("cpu"),
        kind="cpu",
        name=_cpu_name(),
        index=None,
    )


def _cuda_info(index: int) -> DeviceInfo:
    props = torch.cuda.get_device_properties(index)
    return DeviceInfo(
        device=torch.device(f"cuda:{index}"),
        kind="cuda",
        name=props.name,
        index=index,
        total_memory_bytes=props.total_memory,
        capability=(props.major, props.minor),
    )


def _cpu_name() -> str:
    import platform

    return platform.processor() or platform.machine() or "cpu"


def cuda_is_available() -> bool:
    """Explicit, greppable wrapper.

    Use this in `skipif` conditions so that searching for CUDA assumptions finds
    one place rather than scattered `torch.cuda.is_available()` calls.
    """
    return torch.cuda.is_available()


def describe_environment() -> dict:
    """Environment fingerprint for the result record and the smoke test."""
    import platform
    import sys

    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
    }
    try:
        import torch_geometric

        info["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        info["torch_geometric"] = None

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return info


def move_to(obj, device: torch.device):
    """Recursively move tensors in common containers to `device`.

    Leaves non-tensors alone, so it is safe on mixed structures such as a PyG
    ``Data`` carrying a ``raw_texts`` list of strings.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to(v, device) for v in obj]
        return type(obj)(moved) if not isinstance(obj, tuple) else tuple(moved)
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj
