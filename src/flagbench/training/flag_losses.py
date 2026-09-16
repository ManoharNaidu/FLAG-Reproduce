"""The two extra loss terms decision D-001 needs to reproduce `+FLAG*`.

Per D-001 (research/decisions.md), `flag_finetuned` is reproduced as *extra GNN
epochs under the residual + orthogonality regularisation, with the LLM frozen* --
not as LLM fine-tuning, because upstream `train.py`'s gradient path to the LoRA
parameters is severed (`model.generate()` is non-differentiable).

`causal_loss` (upstream `utils.py`) is exactly `F.cross_entropy` and is used
inline in the trainer, so it is not reproduced here.

`non_causal_loss` and `orthogonal_loss` are reproduced below, with one
documented conflict (research/paper_notes.md section 3.5, deviation #5 in
research/reproduction_status.md):

    paper Eq. 9   L_Orthog = || Z_D . Z_R ||_2^2         (squared dot product)
    code          orthogonal_loss = sum(normalize(a) * normalize(b))   (signed cosine)

The squared form is minimised at a dot product of 0 (true orthogonality); the
code's signed cosine is minimised at -1 (anti-alignment) -- not the same
objective. Both are implemented and selectable; the paper's Eq. 9 is the
default, the code's form is available to reproduce the released implementation
bit-for-bit.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

ORTHOGONALITY_MODES = ("squared_dot", "signed_cosine")


def non_causal_loss(non_causal_logits: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """KL divergence of the residual-branch output from a uniform distribution.

    Matches `methods/flag/utils.py:non_causal_loss` exactly: residual text is
    supposed to carry no class signal, so its output is pushed toward uniform.
    """
    uniform = torch.full_like(non_causal_logits, 1.0 / num_classes)
    log_probs = F.log_softmax(non_causal_logits, dim=-1)
    return F.kl_div(log_probs, uniform, reduction="batchmean")


def orthogonal_loss(
    disc_repr: torch.Tensor, common_repr: torch.Tensor, mode: str = "squared_dot",
) -> torch.Tensor:
    """Encourages the discriminative and residual representations to be independent.

    `disc_repr` / `common_repr` are single-example representation vectors (the
    paper's Z_D, Z_R restricted to one node) -- whatever tensor the caller
    chooses, typically the skip-GNN's hidden embedding.
    """
    if mode == "squared_dot":
        return torch.sum(disc_repr * common_repr) ** 2
    if mode == "signed_cosine":
        disc_norm = F.normalize(disc_repr, p=2, dim=-1)
        common_norm = F.normalize(common_repr, p=2, dim=-1)
        return torch.sum(disc_norm * common_norm)
    raise ValueError(f"unknown orthogonality mode {mode!r}; use one of {ORTHOGONALITY_MODES}")
