"""FLAG compatibility check for a text-augmented dataset (brief section 48).

    python -m scripts.test_flag_dataset_compatibility --dataset yelpchi
    python -m scripts.test_flag_dataset_compatibility --dataset amazon --device cuda:0

Steps, in order:
    1. load the dataset through the unified loader
    2. encode node text with the project's SentenceTransformer
    3. build the FLAG-compatible payload through the adapter
    4. instantiate a FLAG backbone via the project's own factory
    5. run ONE forward pass
    6. report the prediction tensor shape

No training. This checks plumbing and alignment, not accuracy.

SCOPE
-----
Text is encoded only for the nodes in the sampled subgraph, because encoding
45,954 reviews on CPU is minutes of work that proves nothing extra for a
plumbing check. `--encode-all` forces the full corpus.

The backbone is built through `flagbench.adapters.backbone.build_backbone`, the
same factory the real training loop uses, so a pass here means the real runner
sees a payload it can consume. No FLAG core file is imported in modified form.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_MODELS = ["gcn", "gat", "bwgnn"]
SBERT = "all-MiniLM-L6-v2"


def induced_subgraph(edge_index: torch.Tensor, nodes: torch.Tensor
                     ) -> torch.Tensor:
    """Edges with both endpoints in `nodes`, re-indexed to 0..len(nodes)-1."""
    n_max = int(edge_index.max()) + 1
    remap = torch.full((n_max,), -1, dtype=torch.long)
    remap[nodes] = torch.arange(len(nodes))
    src, dst = edge_index[0], edge_index[1]
    keep = (remap[src] >= 0) & (remap[dst] >= 0)
    return torch.stack([remap[src[keep]], remap[dst[keep]]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["yelpchi", "amazon"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--num-nodes", type=int, default=512,
                    help="subgraph size for the forward pass")
    ap.add_argument("--encode-all", action="store_true")
    args = ap.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"SKIP: {args.device} requested but CUDA is unavailable. "
              f"A skipped GPU check is never reported as a pass.")
        return 0

    from flagbench.adapters.backbone import build_backbone
    from flagbench.flag_adapter import REGISTRY_KEY, to_flag_payload
    from flagbench.fraud_text import load_dataset, validate_dataset

    print("=" * 78)
    print(f"FLAG COMPATIBILITY — {args.dataset}  (device={args.device})")
    print("=" * 78)
    print(f"  torch {torch.__version__}")

    # 1. load ---------------------------------------------------------------
    ds = load_dataset(args.dataset)
    report = validate_dataset(ds)
    if not report.ok:
        print(report.render())
        return 1
    print(f"\n  [1] loaded        {ds.num_nodes:,} nodes, "
          f"{ds.num_features} features, validation PASS")

    # 2. adapter ------------------------------------------------------------
    payload, manifest = to_flag_payload(ds)
    print(f"  [2] adapter       key='{REGISTRY_KEY[args.dataset]}', "
          f"experiment_type='{manifest['experiment_type']}', "
          f"is_flag_reproduction={manifest['is_flag_reproduction']}")

    # 3. pick a subgraph of nodes that actually have text --------------------
    from flagbench.fraud_text.base import VERIFIED
    with_text = torch.tensor(
        [i for i, s in enumerate(ds.text_status) if s == VERIFIED][:args.num_nodes])
    if len(with_text) == 0:
        print("  no node has VERIFIED text; nothing to encode")
        return 1
    sub_ei = induced_subgraph(payload["edge_index"], with_text)
    print(f"  [3] subgraph      {len(with_text):,} nodes, "
          f"{sub_ei.shape[1]:,} induced edges")

    # 4. text embeddings ----------------------------------------------------
    from sentence_transformers import SentenceTransformer
    targets = (payload["raw_texts"] if args.encode_all
               else [payload["raw_texts"][i] for i in with_text.tolist()])
    encoder = SentenceTransformer(SBERT, device=args.device)
    emb = torch.as_tensor(encoder.encode(targets, batch_size=64,
                                         show_progress_bar=False))
    assert emb.shape[0] == len(targets), "embedding count != text count"
    print(f"  [4] embeddings    {tuple(emb.shape)} via {SBERT}  "
          f"(alignment check: len(texts)==rows OK)")

    # 5. forward pass per backbone -----------------------------------------
    device = torch.device(args.device)
    x = emb.to(device).float()
    ei = sub_ei.to(device)
    ok = True
    print(f"\n  [5] forward passes")
    for model_key in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            net = build_backbone(model_key, in_dim=x.shape[1], out_dim=2,
                                 hidden_dim=32, device=device)
            net.eval()
            with torch.no_grad():
                out = net(x, ei)
            logits = out[1] if isinstance(out, tuple) else out
            shape_ok = logits.shape[0] == x.shape[0] and logits.shape[1] == 2
            finite = bool(torch.isfinite(logits).all())
            status = "OK" if (shape_ok and finite) else "FAIL"
            ok &= shape_ok and finite
            print(f"      {model_key:<10} logits={tuple(logits.shape)} "
                  f"finite={finite}  {status}")
        except Exception as exc:  # noqa: BLE001 - report, never swallow
            ok = False
            print(f"      {model_key:<10} FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:110]}")

    print("\n" + "=" * 78)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} — "
          f"prediction shape ({len(with_text)}, 2) as expected"
          if ok else "RESULT: FAIL")
    print("NOTE: plumbing only. No accuracy claim, and this is a "
          "text-augmented\n      STUDY, never a FLAG reproduction.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
