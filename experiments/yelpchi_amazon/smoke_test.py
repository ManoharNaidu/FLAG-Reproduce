"""Sanity-check a native_<dataset> payload before trusting it for real runs.

Loads data/benchmark/native_<dataset>/graph.pt, checks internal consistency
(no index out of range, masks don't overlap, tensors line up), and runs a
couple of epochs of a plain 2-layer GCN on the 'homo' relation as a forward-
pass smoke test -- NOT a benchmark result. It exists to catch a broken
preprocessing run (e.g. corrupt edge_index) before spending real training
time on it, mirroring the project's "smoke-test before committing" convention
(docs/vastai_gpu_workflow.md).

Usage:
    python -m experiments.yelpchi_amazon.smoke_test --dataset yelpchi
"""
from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

ROOT = pathlib.Path(__file__).resolve().parents[2]


def check_consistency(payload: dict, dataset: str) -> None:
    n = payload["y"].shape[0]
    assert payload["x"].shape[0] == n, "x/y node count mismatch"

    for name, ei in {"homo": payload["edge_index_homo"],
                      **payload["edge_index_relations"]}.items():
        assert ei.shape[0] == 2, f"{name}: edge_index must be [2, E]"
        assert int(ei.max()) < n, f"{name}: edge_index references node >= {n}"
        assert int(ei.min()) >= 0, f"{name}: negative node index"

    masks = ["train_mask", "val_mask", "test_mask", "unlabeled_mask"]
    stacked = torch.stack([payload[m] for m in masks])
    overlap = (stacked.sum(dim=0) > 1).sum().item()
    assert overlap == 0, f"{overlap} nodes appear in more than one of {masks}"

    covered = stacked.sum(dim=0).bool()
    assert covered.all(), (
        f"{(~covered).sum().item()} nodes are in none of {masks} -- "
        "every node must be train, val, test, or explicitly unlabeled"
    )
    print(f"[{dataset}] consistency checks passed ({n:,} nodes)")


def run_forward_smoke(payload: dict, dataset: str, epochs: int = 5) -> None:
    x, y = payload["x"], payload["y"]
    edge_index = payload["edge_index_homo"]
    train_mask, val_mask = payload["train_mask"], payload["val_mask"]

    class TinyGCN(torch.nn.Module):
        def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 2):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, out_dim)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            return self.conv2(h, edge_index)

    torch.manual_seed(0)
    model = TinyGCN(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index)[val_mask].argmax(dim=1)
            val_acc = (pred == y[val_mask]).float().mean().item()
        print(f"[{dataset}] epoch {epoch}: train_loss={loss.item():.4f} "
              f"val_acc={val_acc:.4f}")

    print(
        f"[{dataset}] forward/backward pass completed without error. "
        "This is a plumbing check, NOT a benchmark result -- no tuning, "
        "no seeds averaged, accuracy is misleading on this class imbalance "
        "anyway (see research/degenerate_baselines.md for why accuracy "
        "alone is the wrong metric here)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["yelpchi", "amazon"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    graph_path = ROOT / "data" / "benchmark" / f"native_{args.dataset}" / "graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"{graph_path} not found. Run first:\n"
            f"  python -m experiments.yelpchi_amazon.build_native_benchmark "
            f"--dataset {args.dataset}"
        )

    payload = torch.load(graph_path, map_location="cpu")
    check_consistency(payload, args.dataset)
    run_forward_smoke(payload, args.dataset, epochs=args.epochs)


if __name__ == "__main__":
    main()
