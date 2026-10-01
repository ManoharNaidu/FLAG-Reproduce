"""Shared argparse plumbing so every script builds the SAME SamplingConfig.

The sampling config enters the sampling cache key AND the LLM cache key, so the
scripts that build subgraphs, generate LLM text, encode it and train must all
agree on it exactly. One helper, used by all of them, is how that is guaranteed.
"""
from __future__ import annotations

import argparse

from flagbench.sampling.markov_diffusion import MD_SELECTIONS, OPERATORS
from flagbench.sampling.semantic import SamplingConfig

STRATEGIES = [
    "semantic", "semantic_nothreshold", "random", "none", "feature",
    "markov_diffusion",
]


def add_md_args(parser: argparse.ArgumentParser) -> None:
    """FLAG-MD options. Inert unless --strategy / --sampling-strategy is markov_diffusion."""
    group = parser.add_argument_group("Markov-diffusion sampling (FLAG-MD)")
    group.add_argument("--diffusion-steps", type=int, default=2,
                       help="K in Z(K) = (1/K) sum_{k=0..K} T^k")
    group.add_argument("--diffusion-operator", default="paper_eq6", choices=OPERATORS)
    group.add_argument("--diffusion-threshold", type=float, default=None,
                       help="optional: drop candidates with diffusion distance > tau "
                            "(NOT comparable to the cosine --threshold)")
    group.add_argument("--md-selection", default="matched_cosine", choices=MD_SELECTIONS,
                       help="matched_cosine = keep as many neighbours as cosine keeps "
                            "for the same node (only the ranking differs); "
                            "top_n = keep the N closest, no threshold")


def config_from_args(args, strategy: str | None = None, seed: int = 0) -> SamplingConfig:
    """Build a SamplingConfig from parsed args (falls back to defaults if a
    script does not declare the MD options)."""
    return SamplingConfig(
        hops=args.hops,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        strategy=strategy or args.strategy,
        seed=seed,
        diffusion_steps=getattr(args, "diffusion_steps", 2),
        diffusion_operator=getattr(args, "diffusion_operator", "paper_eq6"),
        diffusion_threshold=getattr(args, "diffusion_threshold", None),
        md_selection=getattr(args, "md_selection", "matched_cosine"),
    )
