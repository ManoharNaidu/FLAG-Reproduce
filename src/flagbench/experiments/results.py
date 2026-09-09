"""Machine-readable result storage. Phase 13.

One JSON file per run under results/raw/, aggregated into
results/aggregated/results.{csv,json}.

Two rules are enforced here rather than left to discipline:

1. **Every row records its provenance.** `impl_source` is required, so a
   `flag_bundled` number can never be silently read as an official baseline
   (decision D-002).
2. **Failures are recorded, not dropped.** A run that crashes still writes a row
   with `status="failed"` and the error. Silently missing rows are how a
   benchmark ends up reporting only the runs that happened to work.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "results" / "raw"
AGG_DIR = ROOT / "results" / "aggregated"

SCHEMA_VERSION = "results-v1"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:
        return "unknown"


@dataclass
class RunResult:
    """One (dataset, model, variant, seed, init) run."""

    # --- identity -------------------------------------------------------
    experiment_id: str
    timestamp: str
    git_commit: str
    schema_version: str = SCHEMA_VERSION

    # --- what was run ---------------------------------------------------
    dataset: str = ""
    model: str = ""
    variant: str = ""
    impl_source: str = ""          # REQUIRED. flag_bundled | official | ...
    model_fidelity: str = ""
    seed: int = 0
    initialization: int = 0

    # --- environment ----------------------------------------------------
    device: str = ""
    device_name: str = ""
    framework: str = ""
    python_version: str = ""
    platform: str = ""

    # --- configuration --------------------------------------------------
    hyperparameters: dict = field(default_factory=dict)
    sampling_config: dict = field(default_factory=dict)
    dataset_version: str = ""
    dataset_manifest_sha256: str = ""
    prompt_version: str = ""
    llm_model: str = ""
    llm_finetuned: bool = False

    # --- outcome --------------------------------------------------------
    best_epoch: int = -1
    epochs_run: int = 0
    stopped_early: bool = False
    threshold: float = float("nan")
    threshold_policy: str = ""

    validation_auc: float = float("nan")
    validation_f1_macro: float = float("nan")
    test_auc: float = float("nan")
    test_f1_macro: float = float("nan")
    test_precision_fraud: float = float("nan")
    test_recall_fraud: float = float("nan")
    test_f1_fraud: float = float("nan")
    test_accuracy: float = float("nan")
    test_ks: float = float("nan")
    test_ece: float = float("nan")

    training_time: float = float("nan")
    inference_time: float = float("nan")
    checkpoint_path: str = ""

    status: str = "completed"      # completed | failed | skipped
    error_message: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def new(cls, **kwargs) -> RunResult:
        return cls(
            experiment_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=git_commit(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            **kwargs,
        )

    def validate(self) -> None:
        if not self.impl_source:
            raise ValueError(
                "impl_source is required on every result row. Without it a "
                "flag_bundled number can be misread as an official baseline "
                "(decision D-002)."
            )
        if not (self.dataset and self.model and self.variant):
            raise ValueError("dataset, model and variant are all required")
        if self.status not in ("completed", "failed", "skipped"):
            raise ValueError(f"unknown status {self.status!r}")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, directory: pathlib.Path | None = None) -> pathlib.Path:
        self.validate()
        directory = pathlib.Path(directory or RAW_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        name = (
            f"{self.dataset}__{self.model}__{self.variant}__"
            f"s{self.seed}i{self.initialization}__{self.experiment_id}.json"
        )
        path = directory / name
        path.write_text(
            json.dumps(self.as_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def summary(self) -> str:
        if self.status != "completed":
            return (
                f"{self.dataset}/{self.model}/{self.variant} "
                f"s{self.seed}i{self.initialization}  {self.status.upper()}: "
                f"{self.error_message[:80]}"
            )
        return (
            f"{self.dataset}/{self.model}/{self.variant} "
            f"s{self.seed}i{self.initialization}  "
            f"test AUC {self.test_auc:.4f}  F1-macro {self.test_f1_macro:.4f}  "
            f"(best epoch {self.best_epoch}, {self.training_time:.1f}s)"
        )


def load_all(directory: pathlib.Path | None = None) -> list[dict]:
    directory = pathlib.Path(directory or RAW_DIR)
    if not directory.exists():
        return []
    rows = []
    for path in sorted(directory.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            print(f"WARNING: could not parse {path.name}: {exc}", file=sys.stderr)
    return rows


FLAT_SKIP = {"hyperparameters", "sampling_config", "extra"}


def aggregate_to_files(
    raw_dir: pathlib.Path | None = None,
    out_dir: pathlib.Path | None = None,
) -> dict:
    """Write results.csv and results.json. Returns a small summary."""
    rows = load_all(raw_dir)
    out_dir = pathlib.Path(out_dir or AGG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        return {"num_runs": 0, "csv": None, "json": None}

    flat = []
    for row in rows:
        item = {k: v for k, v in row.items() if k not in FLAT_SKIP}
        for key in FLAT_SKIP:
            for sub, value in (row.get(key) or {}).items():
                item[f"{key}.{sub}"] = value
        flat.append(item)

    columns: list[str] = []
    for item in flat:
        for key in item:
            if key not in columns:
                columns.append(key)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat)

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8"
    )

    completed = [r for r in rows if r.get("status") == "completed"]
    return {
        "num_runs": len(rows),
        "completed": len(completed),
        "failed": sum(1 for r in rows if r.get("status") == "failed"),
        "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "json": str(json_path.relative_to(ROOT)).replace("\\", "/"),
    }


def build_comparison_table(rows: list[dict] | None = None) -> list[dict]:
    """mean±std per (dataset, model, variant, impl_source), publication-style.

    Grouping **includes impl_source**, so a flag_bundled and an official run of
    the same model never collapse into one cell.
    """
    from flagbench.metrics.classification import aggregate, format_mean_std

    rows = rows if rows is not None else load_all()
    completed = [r for r in rows if r.get("status") == "completed"]

    groups: dict[tuple, list[dict]] = {}
    for row in completed:
        key = (row["dataset"], row["model"], row["variant"], row["impl_source"])
        groups.setdefault(key, []).append(row)

    table = []
    for (dataset, model, variant, impl), members in sorted(groups.items()):
        agg = aggregate(members, keys=["test_f1_macro", "test_auc"])
        table.append({
            "dataset": dataset,
            "model": model,
            "variant": variant,
            "impl_source": impl,
            "n_runs": agg["n_runs"],
            "f1_macro": format_mean_std(
                agg["test_f1_macro_mean"], agg["test_f1_macro_std"]
            ),
            "auc": format_mean_std(agg["test_auc_mean"], agg["test_auc_std"]),
            "f1_macro_mean": agg["test_f1_macro_mean"],
            "auc_mean": agg["test_auc_mean"],
        })
    return table


def render_markdown_table(table: list[dict]) -> str:
    lines = [
        "| Dataset | Model | Variant | Impl | Runs | F1-Macro | AUC |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in table:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['variant']} | "
            f"{row['impl_source']} | {row['n_runs']} | {row['f1_macro']} | "
            f"{row['auc']} |"
        )
    return "\n".join(lines)
