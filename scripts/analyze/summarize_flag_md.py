"""Summarise the FLAG (cosine) vs FLAG-MD (Markov diffusion) comparison.

    python -m scripts.analyze.summarize_flag_md --variant text
    python -m scripts.analyze.summarize_flag_md --variant flag

Reads results/flag_md/<variant>/<sampler>/*.json, writes
results/tables/flag_md_<variant>.{md,json}.

Runs are PAIRED: the (seed, init) streams do not depend on the sampler, so run
(s, i) of cosine and of MD share data order and weight initialisation. Deltas are
therefore computed per pair. Nothing here is a significance claim: the splits are
fixed (seed varies batch order and, with init, weights only), so 8 runs
under-represent true variance.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]

METRICS = [
    ("F1", "test_f1_fraud"), ("F1-macro", "test_f1_macro"), ("AUC", "test_auc"),
    ("Precision", "test_precision_fraud"), ("Recall", "test_recall_fraud"),
]
PRIMARY = ("cosine", "md_K2_matched")


def load(variant: str):
    base = ROOT / "results" / "flag_md" / variant
    data = collections.defaultdict(dict)        # (sampler, ds, model) -> {(seed, init): row}
    dropped = collections.Counter()
    for sdir in sorted(p for p in base.glob("*") if p.is_dir()):
        for f in sdir.glob("*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("status") != "completed":
                dropped[(sdir.name, r["dataset"], r["model"])] += 1
                continue
            key = (sdir.name, r["dataset"], r["model"])
            pair = (r["seed"], r["initialization"])
            if pair in data[key]:
                raise SystemExit(f"duplicate run {key} {pair}: {f}")
            data[key][pair] = r
    return data, dropped


def ms(values):
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}"


def vec(runs, field, pairs=None):
    pairs = pairs if pairs is not None else sorted(runs)
    return np.array([runs[p][field] for p in pairs], dtype=float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="text", choices=["text", "flag", "flag_finetuned"])
    args = ap.parse_args(argv)

    data, dropped = load(args.variant)
    if not data:
        print("no results found")
        return 1
    samplers = sorted({k[0] for k in data})
    datasets = sorted({k[1] for k in data})
    models = sorted({k[2] for k in data})
    out = [f"# FLAG vs FLAG-MD - variant `{args.variant}`\n"]
    summary: dict = {"variant": args.variant, "samplers": samplers}

    if dropped:
        out.append(f"**Failed runs (excluded): {sum(dropped.values())}** {dict(dropped)}\n")

    # ------------------------------------------------------------ per backbone
    a, b = PRIMARY
    for ds in datasets:
        out.append(f"\n## {ds} - `{a}` vs `{b}` (mean ± std over runs)\n")
        out.append("| backbone | sampler | n | " + " | ".join(m for m, _ in METRICS) + " |")
        out.append("|---|---|---|" + "---|" * len(METRICS))
        deltas = {m: [] for m, _ in METRICS}
        for model in models:
            ra, rb = data.get((a, ds, model)), data.get((b, ds, model))
            if not ra or not rb:
                continue
            pairs = sorted(set(ra) & set(rb))
            for name, runs in ((a, ra), (b, rb)):
                out.append(f"| {model} | {name} | {len(pairs)} | " + " | ".join(
                    ms(vec(runs, f, pairs)) for _, f in METRICS) + " |")
            for m, f in METRICS:
                deltas[m].append(float((vec(rb, f, pairs) - vec(ra, f, pairs)).mean()))
        out.append("")
        out.append(f"**Paired delta (`{b}` - `{a}`), per-backbone mean over the same (seed, init) pairs:**\n")
        out.append("| metric | mean delta over backbones | backbones with delta > 0 | min | max |")
        out.append("|---|---|---|---|---|")
        for m, _ in METRICS:
            d = np.array(deltas[m])
            if d.size:
                out.append(f"| {m} | {d.mean():+.4f} | {int((d > 0).sum())}/{d.size} | "
                           f"{d.min():+.4f} | {d.max():+.4f} |")
        summary[f"{ds}_paired_delta"] = {m: deltas[m] for m, _ in METRICS}

    # --------------------------------------------------------------- K ablation
    kk = sorted([s for s in samplers if s.startswith("md_K") and s.endswith("_matched")],
                key=lambda s: int(s[4:].split("_")[0]))
    if len(kk) > 1:
        out.append("\n## Diffusion depth K (matched budget): mean over backbones ± std across backbones\n")
        out.append("| dataset | K | F1 | F1-macro | AUC |")
        out.append("|---|---|---|---|---|")
        for ds in datasets:
            for s in ["cosine"] + kk:
                per = []
                for model in models:
                    runs = data.get((s, ds, model))
                    if runs:
                        per.append([vec(runs, f).mean() for _, f in METRICS[:3]])
                if per:
                    per = np.array(per)
                    label = "cosine (ref)" if s == "cosine" else s[4:].split("_")[0]
                    out.append(f"| {ds} | {label} | " + " | ".join(
                        f"{per[:, i].mean():.4f} ± {per[:, i].std(ddof=1) if len(per) > 1 else 0:.4f}"
                        for i in range(3)) + " |")

    # -------------------------------------------------------- selection mode
    if "md_K2_top_n" in samplers:
        out.append("\n## Selection rule at K=2: matched budget vs literal top-N (no threshold)\n")
        out.append("| dataset | sampler | F1 | F1-macro | AUC |")
        out.append("|---|---|---|---|---|")
        for ds in datasets:
            for s in ("cosine", "md_K2_matched", "md_K2_top_n"):
                per = np.array([[vec(data[(s, ds, m)], f).mean() for _, f in METRICS[:3]]
                                for m in models if (s, ds, m) in data])
                if per.size:
                    out.append(f"| {ds} | {s} | " + " | ".join(f"{per[:, i].mean():.4f}" for i in range(3)) + " |")

    # ------------------------------------------------------------------ runtime
    out.append("\n## Runtime and memory (mean per run)\n")
    out.append("| sampler | train s | inference s | sampling s (one-off, per dataset) | H=Z(K)X precompute s | peak GPU MB |")
    out.append("|---|---|---|---|---|---|")
    for s in samplers:
        rows = [r for k, runs in data.items() if k[0] == s for r in runs.values()]

        def avg(getter):
            v = [getter(r) for r in rows]
            v = [x for x in v if isinstance(x, (int, float))]
            return f"{np.mean(v):.2f}" if v else "n/a"

        out.append(f"| {s} | {avg(lambda r: r.get('training_time'))} | {avg(lambda r: r.get('inference_time'))} | "
                   f"{avg(lambda r: r['extra'].get('sampling_seconds'))} | "
                   f"{avg(lambda r: r['extra'].get('diffusion_precompute_seconds'))} | "
                   f"{avg(lambda r: r['extra'].get('peak_gpu_mem_mb'))} |")
    out.append("\n_Sampling seconds for matched-budget MD include evaluating cosine to size each budget; "
               "`md_K2_top_n` is the pure MD selection cost._")

    text = "\n".join(out) + "\n"
    dest = ROOT / "results" / "tables" / f"flag_md_{args.variant}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    dest.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(text)
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
