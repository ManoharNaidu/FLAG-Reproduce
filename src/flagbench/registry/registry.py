"""Model / variant / dataset registry with capability validation.

Phases 24 and 41. The registry's job is to **refuse impossible experiments
loudly** instead of silently substituting something that runs.

The rule that matters most (Phase 9): the FLAG paper itself says Yelp-Fraud,
Amazon-Fraud, T-Finance and T-Social "lack textual information". So a FLAG
variant on those datasets is not a hard experiment, it is a meaningless one, and
`validate()` rejects it rather than fabricating text.

Two implementation lineages are carried per backbone and are never merged
(decision D-002):

    flag_bundled : the FLAG authors' own PyG rewrite in methods/flag/.
                   Reproduces THE PAPER'S NUMBERS.
    official     : the baseline as its own authors published it.
                   Reproduces THE BASELINE.

Four of the five bundled baselines are not their published algorithms, so
reporting one as the other would misattribute performance. `impl_source` is a
required field on every result row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ImplSource(str, Enum):
    """WHERE an implementation came from. Stamped into every result row.

    This is provenance, NOT a quality judgement -- keep the two separate. An
    earlier version used APPROXIMATION here, which silently excluded CARE-GNN,
    BWGNN and DGA-GNN from the comparison against the paper: they ARE the
    lineage that produced Table 4, however unfaithful they are to their own
    published algorithms. How faithful a model is lives in `Fidelity`.
    """

    OFFICIAL = "official"
    """The baseline's own authors' code."""

    FLAG_BUNDLED = "flag_bundled"
    """FLAG's own rewrite in methods/flag/. This is what produced Table 4."""

    REIMPLEMENTED = "reimplemented"
    """Written here from the paper's equations."""

    UNKNOWN = "unknown"


class Fidelity(str, Enum):
    """HOW CLOSE an implementation is to the algorithm it is named after.

    Orthogonal to ImplSource. A flag_bundled model can be FAITHFUL (GCN) or an
    APPROXIMATION (CARE-GNN); an official one is FAITHFUL by definition.
    """

    FAITHFUL = "faithful"
    ADAPTED = "adapted"
    APPROXIMATION = "approximation"
    """Missing components that define the published algorithm. Must never be
    reported under the bare model name without qualification."""
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Capabilities:
    """What a method can and cannot do. Phase 41."""

    homogeneous_graph: bool = True
    heterogeneous_graph: bool = False
    multi_relation: bool = False
    requires_multi_relation: bool = False
    requires_native_text: bool = False
    inductive: bool = True
    cpu: bool = True
    gpu: bool = True
    mini_batch: bool = True
    node_classification: bool = True

    def as_record(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    impl_source: ImplSource
    module: str
    """Import path of the class, e.g. 'models:GCN' inside methods/flag."""
    fidelity: str
    """Honest prose assessment vs the published algorithm."""
    fidelity_class: "Fidelity" = None  # type: ignore[assignment]
    """Machine-readable counterpart of `fidelity`. Set in __post_init__ if omitted."""
    capabilities: Capabilities = field(default_factory=Capabilities)
    hidden_dim_locked: int | None = None
    """Some bundled backbones only work at one hidden size (DGA, PMP bind `x32`
    behind `if len(x[0]) == 32`). None means any size."""
    has_skip: bool = False
    """Whether upstream applies the paper's Eq. 6 skip to this backbone."""
    notes: str = ""

    def as_record(self) -> dict:
        return {
            "model": self.key,
            "model_display_name": self.display_name,
            "impl_source": self.impl_source.value,
            "fidelity_class": (
                self.fidelity_class.value if self.fidelity_class
                else Fidelity.UNKNOWN.value
            ),
            "model_fidelity": self.fidelity,
            "model_has_skip": self.has_skip,
        }


@dataclass(frozen=True)
class VariantSpec:
    key: str
    display_name: str
    feature_source: str
    """Which node features feed the GNN:
        stored      -- the graph's own `x` (4096-d, Llama-2-derived)
        lm_text     -- Sentence-BERT of the RAW node text (384-d)
        lm_disc     -- Sentence-BERT of the LLM's DISCRIMINATIVE text (384-d)
    """
    requires_llm: bool = False
    requires_finetuned_llm: bool = False
    dual_branch: bool = False
    """Whether inference fuses raw-text and discriminative-text branches through
    the paper's attention layer (models.py:DualGNN)."""

    default_sampling_strategy: str = "none"
    """Which neighbour-sampling strategy this variant uses by default.

    Semantic similarity sampling (SS) is a FLAG *contribution*, not part of the
    evaluation protocol: the paper's Table 5 ablates it as a component alongside
    LLM and SG. So `baseline` and `+text` must use plain 2-hop neighbourhoods
    (NS), and only the FLAG variants get SS. Giving the baseline SS would hand it
    part of the method it is supposed to be a baseline for, and would overstate
    the baseline while understating FLAG's contribution.

    The paper's "we evaluate the models on 2-hop subgraphs" applies to all
    variants; the *sampling* on top of that is what differs.
    """
    notes: str = ""

    def as_record(self) -> dict:
        return {
            "variant": self.key,
            "variant_display_name": self.display_name,
            "feature_source": self.feature_source,
            "requires_llm": self.requires_llm,
            "default_sampling_strategy": self.default_sampling_strategy,
        }


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    has_native_text: bool
    num_relations: int
    source: str
    obtainable: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Models. Fidelity strings come from research/flag_code_audit.md section 7,
# each backed by a passing test in tests/integration/test_flag_upstream_claims.py.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gcn": ModelSpec(
        key="gcn", fidelity_class=Fidelity.FAITHFUL, display_name="GCN", impl_source=ImplSource.FLAG_BUNDLED,
        module="models:GCN",
        fidelity="faithful - two GCNConv layers plus the Eq. 6 skip",
        has_skip=True,
    ),
    "gat": ModelSpec(
        key="gat", fidelity_class=Fidelity.ADAPTED, display_name="GAT (FLAG's variant)",
        impl_source=ImplSource.FLAG_BUNDLED, module="models:GAT",
        fidelity=(
            "NOT canonical GAT - conv2 is a SAGEConv, not a GATConv, so the "
            "second layer performs no attention"
        ),
        has_skip=False,
        notes="initial_x is computed and discarded; no skip despite Eq. 6",
    ),
    "geniepath": ModelSpec(
        key="geniepath", fidelity_class=Fidelity.ADAPTED, display_name="GeniePath",
        impl_source=ImplSource.FLAG_BUNDLED, module="geniepath:GeniePathLazy",
        fidelity=(
            "closest to faithful - Breadth(GAT)+Depth(LSTM) preserved; matches "
            "PyG's example globals exactly. No official author repo exists."
        ),
        has_skip=True,
        notes="4 layers and dim=256 as module globals, contradicting the "
              "paper's 'two layers, hidden 64'",
    ),
    "care_gnn": ModelSpec(
        key="care_gnn", fidelity_class=Fidelity.APPROXIMATION, display_name="CARE-GNN (FLAG's variant)",
        impl_source=ImplSource.FLAG_BUNDLED, module="caregnn:CAREGNN",
        fidelity=(
            "NOT CARE-GNN - no RL neighbour filtering, no label-aware "
            "similarity, no multi-relation aggregation. A similarity-gated "
            "mean aggregator."
        ),
        has_skip=False,
    ),
    "bwgnn": ModelSpec(
        key="bwgnn", fidelity_class=Fidelity.APPROXIMATION, display_name="BWGNN (FLAG's variant)",
        impl_source=ImplSource.FLAG_BUNDLED, module="bwgnn:BWGNN",
        fidelity=(
            "beta-wavelet basis NOT applied - the polynomial is evaluated over "
            "raw adjacency A, not the normalised Laplacian, so it is not a "
            "beta wavelet. Theta coefficients themselves are correct."
        ),
        has_skip=True,
    ),
    "dga_gnn": ModelSpec(
        key="dga_gnn", fidelity_class=Fidelity.APPROXIMATION, display_name="DGA-GNN (FLAG's variant)",
        impl_source=ImplSource.FLAG_BUNDLED, module="dga:DGA",
        fidelity=(
            "NOT DGA-GNN - no decision-tree dynamic grouping, no bidirectional "
            "grouped aggregation. GraphSAGE-mean."
        ),
        has_skip=False, hidden_dim_locked=32,
        notes="raises UnboundLocalError at any hidden size but 32",
    ),
    "pmp": ModelSpec(
        key="pmp", fidelity_class=Fidelity.ADAPTED, display_name="PMP (FLAG's variant)",
        impl_source=ImplSource.FLAG_BUNDLED, module="pmp:LASAGE_S",
        fidelity=(
            "partial - fraud/benign partition present, but applied to the "
            "aggregated mean rather than per-neighbour before aggregation"
        ),
        has_skip=False, hidden_dim_locked=32,
        notes="raises UnboundLocalError at any hidden size but 32",
    ),
}


# ---------------------------------------------------------------------------
# Variants. Phase 5 -- these are four distinct experiments, never collapsed.
# ---------------------------------------------------------------------------
VARIANT_REGISTRY: dict[str, VariantSpec] = {
    "baseline": VariantSpec(
        key="baseline", display_name="baseline", feature_source="stored",
        notes=(
            "the paper calls these 'shallow embeddings', but the stored "
            "features are 4096-d and Llama-2-derived, so the label is "
            "questionable - see research/dataset_notes.md section 7"
        ),
    ),
    "text": VariantSpec(
        key="text", display_name="+text", feature_source="lm_text",
        notes="Sentence-BERT of the raw node text (384-d)",
    ),
    "flag": VariantSpec(
        key="flag", display_name="+FLAG", feature_source="lm_disc",
        requires_llm=True, dual_branch=True,
        default_sampling_strategy="semantic",
        notes="zero-shot: LLM extracts discriminative text; attention-fused "
              "with the raw-text branch (models.py:DualGNN)",
    ),
    "flag_finetuned": VariantSpec(
        key="flag_finetuned", display_name="+FLAG*", feature_source="lm_disc",
        requires_llm=True, requires_finetuned_llm=True, dual_branch=True,
        default_sampling_strategy="semantic",
        notes=(
            "decision D-001: upstream's LoRA gradient path is severed, so this "
            "reproduces as extra GNN epochs under the residual+orthogonality "
            "losses with the LLM frozen. Result rows carry llm_finetuned=false."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Datasets. `has_native_text` is the gate for the FLAG variants, and the FLAG
# paper itself is the authority for the False rows.
# ---------------------------------------------------------------------------
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "reddit": DatasetSpec(
        key="reddit", display_name="Reddit", has_native_text=True,
        num_relations=1, source="GLBench",
    ),
    "instagram": DatasetSpec(
        key="instagram", display_name="Instagram", has_native_text=True,
        num_relations=1, source="GLBench",
    ),
    "yelpchi": DatasetSpec(
        key="yelpchi", display_name="YelpChi", has_native_text=False,
        num_relations=3, source="DGL FraudYelpDataset / CARE-GNN",
        notes="the FLAG paper states it lacks textual information",
    ),
    "amazon": DatasetSpec(
        key="amazon", display_name="Amazon", has_native_text=False,
        num_relations=3, source="DGL FraudAmazonDataset / CARE-GNN",
        notes="the FLAG paper states it lacks textual information",
    ),
    "tfinance": DatasetSpec(
        key="tfinance", display_name="T-Finance", has_native_text=False,
        num_relations=1, source="BWGNN Google Drive",
    ),
    "tsocial": DatasetSpec(
        key="tsocial", display_name="T-Social", has_native_text=False,
        num_relations=1, source="BWGNN Google Drive",
    ),
    "elliptic": DatasetSpec(
        key="elliptic", display_name="Elliptic", has_native_text=False,
        num_relations=1, source="DGA-GNN Google Drive",
    ),
    "huabei": DatasetSpec(
        key="huabei", display_name="Huabei (industrial)", has_native_text=True,
        num_relations=1, source="proprietary Alipay", obtainable=False,
        notes="13M nodes / 120M edges. NOT OBTAINABLE outside Alipay; Table 3 "
              "can never be reproduced externally.",
    ),
}


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    def report(self) -> str:
        lines = ["VALID" if self.ok else "NOT AVAILABLE"]
        for reason in self.reasons:
            lines.append(f"  Reason: {reason}")
        if self.suggestions:
            lines.append("  Suggested valid experiments:")
            lines.extend(f"    - {s}" for s in self.suggestions)
        return "\n".join(lines)


class RegistryError(KeyError):
    pass


def get_model(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise RegistryError(
            f"unknown model {key!r}; registered: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key]


def get_variant(key: str) -> VariantSpec:
    if key not in VARIANT_REGISTRY:
        raise RegistryError(
            f"unknown variant {key!r}; registered: {sorted(VARIANT_REGISTRY)}"
        )
    return VARIANT_REGISTRY[key]


def get_dataset(key: str) -> DatasetSpec:
    if key not in DATASET_REGISTRY:
        raise RegistryError(
            f"unknown dataset {key!r}; registered: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[key]


def validate(
    dataset: str,
    model: str,
    variant: str = "baseline",
    device: str = "cpu",
    hidden_dim: int | None = None,
) -> ValidationResult:
    """Decide whether an experiment is meaningful, and say why if not.

    Never silently rewrites the request. A caller that gets `ok=False` should
    stop, not fall back.
    """
    reasons: list[str] = []
    suggestions: list[str] = []

    ds = get_dataset(dataset)
    ms = get_model(model)
    vs = get_variant(variant)

    if not ds.obtainable:
        reasons.append(
            f"{ds.display_name} is not obtainable ({ds.source}). {ds.notes}"
        )

    # --- Phase 9: the integrity rule -----------------------------------
    if vs.requires_llm and not ds.has_native_text:
        reasons.append(
            f"FLAG canonical text mode: NOT AVAILABLE. {ds.display_name} has no "
            f"native text. {ds.notes or ''}".strip()
        )
        suggestions += [
            f"--dataset {dataset} --variant baseline",
            f"--dataset {dataset} --model <fraud GNN> --variant baseline "
            f"(native fraud methods)",
            "a text_augmented_experiment, ONLY with an explicitly provided "
            "text source, recorded as native_text=false and never labelled FLAG",
        ]

    if vs.feature_source == "lm_text" and not ds.has_native_text:
        reasons.append(
            f"variant '+text' needs raw node text, which {ds.display_name} "
            f"does not have"
        )

    # --- capability checks ---------------------------------------------
    if ms.capabilities.requires_multi_relation and ds.num_relations < 2:
        reasons.append(
            f"{ms.display_name} requires a multi-relation graph; "
            f"{ds.display_name} has {ds.num_relations}"
        )

    if device.startswith("cuda") and not ms.capabilities.gpu:
        reasons.append(f"{ms.display_name} does not support GPU execution")
    if device == "cpu" and not ms.capabilities.cpu:
        reasons.append(f"{ms.display_name} does not support CPU execution")

    if (
        hidden_dim is not None
        and ms.hidden_dim_locked is not None
        and hidden_dim != ms.hidden_dim_locked
    ):
        reasons.append(
            f"{ms.display_name} only works at hidden_dim="
            f"{ms.hidden_dim_locked} ({ms.notes})"
        )
        suggestions.append(f"--hidden-dim {ms.hidden_dim_locked}")

    return ValidationResult(ok=not reasons, reasons=reasons, suggestions=suggestions)


def compatibility_matrix() -> list[dict]:
    """Every (dataset, model, variant) triple with its verdict. Phase 24."""
    rows = []
    for dataset in DATASET_REGISTRY:
        for model in MODEL_REGISTRY:
            for variant in VARIANT_REGISTRY:
                result = validate(dataset, model, variant)
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "variant": variant,
                    "valid": result.ok,
                    "reasons": result.reasons,
                })
    return rows
