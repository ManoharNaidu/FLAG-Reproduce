"""Regenerate the auto-generated sections of the main-run report.

    python -m scripts.analyze.build_main_run_report            # in place
    python -m scripts.analyze.build_main_run_report --check    # print, do not write

Rewrites only the text between `<!-- AUTO:<name> BEGIN -->` and
`<!-- AUTO:<name> END -->` markers in the report, so the hand-written parts are
never touched. Reads every result JSON under results/main/ and results/main_gpu/
(the latter supersedes the former for the same (seed, init), matching the
dashboard), the vLLM cache manifests, and the previous low-coverage run under
results/flag_md/ for the before/after comparison.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import pathlib
import re
import statistics as st

import numpy as np
from scipy.stats import wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT = ROOT / "results" / "2026-10-02-flag-cosine-vs-md-main-run-report.md"
DATASETS = ["reddit", "instagram", "amazon_text", "yelpchi_text"]
DS = {"reddit": "Reddit", "instagram": "Instagram", "amazon_text": "Amazon", "yelpchi_text": "YelpChi"}
MODELS = ["bwgnn", "care_gnn", "dga_gnn", "gat", "gcn", "geniepath", "pmp"]
GROUPS = [("baseline", "default"), ("flag", "cosine"), ("flag", "md_K2_matched"),
          ("flag_finetuned", "cosine"), ("flag_finetuned", "md_K2_matched")]
GNAME = {("baseline", "default"): "baseline", ("flag", "cosine"): "flag · cosine",
         ("flag", "md_K2_matched"): "flag · FLAG-MD", ("flag_finetuned", "cosine"): "flag_finetuned · cosine",
         ("flag_finetuned", "md_K2_matched"): "flag_finetuned · FLAG-MD"}
RX = re.compile(r"__s(\d+)i(\d+)__")
PER_CELL = 25


def load_runs():
    """{(variant, sampler, dataset, model): {(seed, init): record}}"""
    out = {}
    for root in ("results/main", "results/main_gpu"):          # main_gpu read last: supersedes
        for f in glob.glob(str(ROOT / root / "*" / "*" / "*.json")):
            parts = pathlib.Path(f).parts
            v, s = parts[-3], parts[-2]
            ds, m = pathlib.Path(f).name.split("__")[:2]
            mm = RX.search(f)
            try:
                r = json.load(open(f))
            except (OSError, ValueError):
                continue
            if r.get("status") != "completed" or not mm:
                continue
            try:
                wall = os.path.getmtime(f) - dt.datetime.fromisoformat(r["timestamp"]).timestamp()
            except (KeyError, ValueError):
                wall = None
            out.setdefault((v, s, ds, m), {})[tuple(map(int, mm.groups()))] = {
                "f1": r["test_f1_macro"], "auc": r["test_auc"], "wall": wall,
                "device": "gpu" if "main_gpu" in f or str(r.get("device", "")).startswith("cuda") else "cpu"}
    return out


def ms(xs, digits=3):
    if not xs:
        return "—"
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    return f"{st.fmean(xs):.{digits}f} ± {sd:.{digits}f}"


def sign(x):
    r = f"{abs(x):.3f}"
    return ("±" if r == "0.000" else "+" if x > 0 else "−") + r


def block(name, body):
    return f"<!-- AUTO:{name} BEGIN -->\n{body.rstrip()}\n<!-- AUTO:{name} END -->"


def sec_progress(runs):
    lines = ["| Variant · sampler | " + " | ".join(DS[d] for d in DATASETS) + " | Total |",
             "|---|" + "---:|" * (len(DATASETS) + 1)]
    grand = 0
    for g in GROUPS:
        row, tot = [], 0
        for d in DATASETS:
            n = sum(min(len(runs.get((*g, d, m), {})), PER_CELL) for m in MODELS)
            tot += n
            row.append(f"{n}/{PER_CELL * len(MODELS)}")
        grand += tot
        lines.append(f"| {GNAME[g]} | " + " | ".join(row) + f" | **{tot}/700** |")
    lines.append(f"| **All** | | | | | **{grand}/3500 ({100 * grand / 3500:.1f}%)** |")
    open_cells = [(g, d, m, len(runs.get((*g, d, m), {}))) for g in GROUPS for d in DATASETS for m in MODELS
                  if len(runs.get((*g, d, m), {})) < PER_CELL]
    status = ("**Complete: every cell has 25/25 runs.**" if not open_cells else
              f"**In progress:** {len(open_cells)} cells below 25/25: " +
              ", ".join(f"{GNAME[g]} / {DS[d]} / {m} ({n}/25)" for g, d, m, n in open_cells) + ".")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"_Generated {stamp}._\n\n" + "\n".join(lines) + "\n\n" + status


def sec_results(runs, metric):
    out = []
    for d in DATASETS:
        out += [f"**{DS[d]}** ({'F1-macro' if metric == 'f1' else 'AUC'}, mean ± std over runs; "
                f"n shown when below 25)", "",
                "| Backbone | " + " | ".join(GNAME[g] for g in GROUPS) + " |",
                "|---|" + "---:|" * len(GROUPS)]
        for m in MODELS:
            cells = []
            for g in GROUPS:
                rs = runs.get((*g, d, m), {})
                xs = [r[metric] for r in rs.values()]
                cells.append(ms(xs) + ("" if len(xs) >= PER_CELL else f" (n={len(xs)})"))
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def paired(runs, variant, d, m):
    a = runs.get((variant, "cosine", d, m), {})
    b = runs.get((variant, "md_K2_matched", d, m), {})
    keys = sorted(set(a) & set(b))
    complete = len(a) >= PER_CELL and len(b) >= PER_CELL
    res = {"complete": complete, "n_pairs": len(keys), "na": len(a), "nb": len(b)}
    for k in ("f1", "auc"):
        xa = [a[x][k] for x in a]; xb = [b[x][k] for x in b]
        if not xa or not xb:
            res[k] = None; continue
        dlt = st.fmean(xb) - st.fmean(xa)
        noise = max(st.stdev(xa) if len(xa) > 1 else 0, st.stdev(xb) if len(xb) > 1 else 0)
        cls = "tie" if abs(dlt) < noise else "win" if dlt > 0 else "loss"
        p = None
        if len(keys) >= 6:
            diffs = np.array([b[x][k] - a[x][k] for x in keys])
            if np.any(diffs != 0):
                p = float(wilcoxon(diffs).pvalue)
        res[k] = {"d": dlt, "cls": cls, "p": p}
    return res


def sec_compare(runs):
    rows = {(v, d, m): paired(runs, v, d, m) for v in ("flag", "flag_finetuned") for d in DATASETS for m in MODELS}
    comp = {k: r for k, r in rows.items() if r["complete"] and r["f1"] and r["auc"]}

    def summ(sel):
        xs = [r for k, r in comp.items() if sel(k)]
        if not xs:
            return None
        o = {"n": len(xs)}
        for k in ("f1", "auc"):
            o[k] = st.fmean(r[k]["d"] for r in xs)
            o[k + "w"] = tuple(sum(r[k]["cls"] == c for r in xs) for c in ("win", "loss", "tie"))
            o[k + "sig"] = (sum(1 for r in xs if r[k]["p"] is not None and r[k]["p"] < 0.05 and r[k]["d"] > 0),
                            sum(1 for r in xs if r[k]["p"] is not None and r[k]["p"] < 0.05 and r[k]["d"] < 0))
        return o

    lines = ["Δ = FLAG-MD − cosine (positive = FLAG-MD better). W/L/T uses the dashboard rule: tie when "
             "|Δ| < the larger of the two stds. *sig+ / sig−* = cells where a paired Wilcoxon signed-rank "
             "test over the matched (seed, init) runs gives p < 0.05 in that direction (no multiple-comparison "
             "correction; read as indicative). Partial cells are excluded.", "",
             "| Slice | pairs | Avg Δ F1 | F1 W/L/T | F1 sig+/sig− | Avg Δ AUC | AUC W/L/T | AUC sig+/sig− |",
             "|---|---:|---:|---|---|---:|---|---|"]
    slices = [("**All**", lambda k: True), ("flag", lambda k: k[0] == "flag"),
              ("flag_finetuned", lambda k: k[0] == "flag_finetuned")]
    slices += [(DS[d], lambda k, d=d: k[1] == d) for d in DATASETS]
    slices += [(m, lambda k, m=m: k[2] == m) for m in MODELS]
    for name, sel in slices:
        o = summ(sel)
        if not o:
            lines.append(f"| {name} | 0 | — | — | — | — | — | — |"); continue
        lines.append(f"| {name} | {o['n']} | {sign(o['f1'])} | {'/'.join(map(str, o['f1w']))} | "
                     f"{o['f1sig'][0]}/{o['f1sig'][1]} | {sign(o['auc'])} | {'/'.join(map(str, o['aucw']))} | "
                     f"{o['aucsig'][0]}/{o['aucsig'][1]} |")
    partial = [k for k, r in rows.items() if not r["complete"]]
    lines += ["", f"Complete pairs: **{len(comp)} of {len(rows)}**." +
              (f" Partial (excluded): " + ", ".join(f"{v} / {DS[d]} / {m} ({rows[(v, d, m)]['na']}·{rows[(v, d, m)]['nb']})"
                                                    for v, d, m in partial) + "." if partial else ""), "",
              "Per-cell detail (complete pairs; ✓ = paired p < 0.05):", "",
              "| Variant | Dataset | Backbone | Δ F1 | p(F1) | Δ AUC | p(AUC) |", "|---|---|---|---:|---:|---:|---:|"]
    for (v, d, m), r in sorted(comp.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        fp = r["f1"]["p"]; ap = r["auc"]["p"]
        lines.append(f"| {v} | {DS[d]} | {m} | {sign(r['f1']['d'])} | "
                     f"{'—' if fp is None else f'{fp:.3f}' + (' ✓' if fp < 0.05 else '')} | {sign(r['auc']['d'])} | "
                     f"{'—' if ap is None else f'{ap:.3f}' + (' ✓' if ap < 0.05 else '')} |")
    return "\n".join(lines)


def sec_vs_baseline(runs):
    lines = ["Average over the 7 backbones of (variant mean − baseline mean), complete cells only.", "",
             "| Dataset | metric | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |",
             "|---|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        for k, name in (("f1", "F1-macro"), ("auc", "AUC")):
            cells = []
            for g in GROUPS[1:]:
                ds_ = []
                for m in MODELS:
                    b = runs.get(("baseline", "default", d, m), {}); x = runs.get((*g, d, m), {})
                    if len(b) >= PER_CELL and len(x) >= PER_CELL:
                        ds_.append(st.fmean(r[k] for r in x.values()) - st.fmean(r[k] for r in b.values()))
                cells.append(f"{sign(st.fmean(ds_))} ({len(ds_)}/7)" if ds_ else "—")
            lines.append(f"| {DS[d]} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sec_before_after(runs):
    old, text = {}, {}
    for variant, store in (("flag", old), ("text", text)):
        for f in glob.glob(str(ROOT / f"results/flag_md/{variant}/*/*.json")):
            s = pathlib.Path(f).parts[-2]
            if s not in ("cosine", "md_K2_matched"):
                continue
            ds, m = pathlib.Path(f).name.split("__")[:2]
            try:
                r = json.load(open(f))
            except (OSError, ValueError):
                continue
            if r.get("status") == "completed":
                store.setdefault((s, ds, m), []).append(r["test_f1_macro"])
    lines = ["flag variant, F1-macro averaged over the 7 backbones. *Previous* = results/flag_md/flag "
             "(64-token / 300-char LLM budget, 0.6–12.9% node coverage, 4 seeds × 2 inits). "
             "*This run* = results/main (550 / 1200 budget, vLLM, 5 × 5). *Raw text only* = the previous "
             "run's `text` variant (Sentence-BERT of the raw node text, same sampler, no LLM text, 4 × 2), "
             "shown as a reference for how much the LLM branch adds.", "",
             "| Dataset | Sampler | Raw text only (prev.) | flag, previous | flag, this run | Δ (this − previous) |",
             "|---|---|---:|---:|---:|---:|"]
    for d in ("reddit", "instagram"):
        for s in ("cosine", "md_K2_matched"):
            o = [st.fmean(old[(s, d, m)]) for m in MODELS if old.get((s, d, m))]
            n = [st.fmean(r["f1"] for r in runs[("flag", s, d, m)].values()) for m in MODELS
                 if len(runs.get(("flag", s, d, m), {})) >= PER_CELL]
            t = [st.fmean(text[(s, d, m)]) for m in MODELS if text.get((s, d, m))]
            if o and n:
                lines.append(f"| {DS[d]} | {'cosine' if s == 'cosine' else 'FLAG-MD'} | "
                             f"{f'{st.fmean(t):.3f}' if t else '—'} | {st.fmean(o):.3f} | "
                             f"{st.fmean(n):.3f} | {sign(st.fmean(n) - st.fmean(o))} |")
    return "\n".join(lines)


def sec_coverage():
    rows = {}
    for f in glob.glob(str(ROOT / "cache/llm/*.manifest.json")):
        m = json.load(open(f))
        if m.get("llm_config", {}).get("engine") != "vllm":
            continue
        s = "cosine" if m["sampling_config"]["strategy"] == "semantic" else "FLAG-MD"
        rows[(m["dataset"], m["kind"], s)] = m
    lines = ["| Dataset | Kind | Sampler | top-k | Subgraphs | Succeeded | Node coverage | Context overflows | "
             "Reused from cosine | GPU time (s, max shard) |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        for k in ("discriminative", "residual"):
            for s in ("cosine", "FLAG-MD"):
                m = rows.get((d, k, s))
                if not m:
                    continue
                stt = m["stats"]; r = m.get("reuse_from_cosine") or {}
                lines.append(f"| {DS[d]} | {k} | {s} | {m['sampling_config']['top_k']} | {stt['total_subgraphs']:,} | "
                             f"{stt['succeeded']:,} ({100 * stt['subgraph_success_rate']:.1f}%) | "
                             f"{100 * stt['node_coverage']:.1f}% | {stt.get('context_overflows', 0)} | "
                             f"{r.get('reused_texts', '—') if r else '—'} | {stt['seconds']:.0f} |")
    return "\n".join(lines)


def sec_timing(runs):
    lines = ["Median wall-clock minutes per run (result timestamp → file written), over all devices.", "",
             "| Dataset | Backbone | baseline | flag (cos/MD) | flag_finetuned (cos/MD) |", "|---|---|---:|---:|---:|"]
    for d in DATASETS:
        for m in MODELS:
            def med(g):
                w = [r["wall"] for r in runs.get((*g, d, m), {}).values() if r["wall"]]
                return f"{st.median(w) / 60:.0f}" if w else "—"
            lines.append(f"| {DS[d]} | {m} | {med(GROUPS[0])} | {med(GROUPS[1])} / {med(GROUPS[2])} | "
                         f"{med(GROUPS[3])} / {med(GROUPS[4])} |")
    return "\n".join(lines)


def _avg(runs, key, metric):
    xs = runs.get(key, {})
    return st.fmean(r[metric] for r in xs.values()) if xs else None


def sec_feat(runs):
    """flag_feat (D-006) vs baseline and flag on the text-augmented datasets."""
    cols = [("baseline", "default"), ("flag", "cosine"), ("flag", "md_K2_matched"),
            ("flag_feat", "cosine"), ("flag_feat", "md_K2_matched")]
    names = ["baseline", "flag · cosine", "flag · FLAG-MD", "flag_feat · cosine", "flag_feat · FLAG-MD"]
    out = []
    for metric, label in (("f1", "F1-macro"), ("auc", "AUC")):
        for d in ("amazon_text", "yelpchi_text"):
            out += [f"**{DS[d]}, {label}** (mean ± std, 25 runs; n shown when below 25)", "",
                    "| Backbone | " + " | ".join(names) + " |", "|---|" + "---:|" * len(cols)]
            for m in MODELS:
                cells = []
                for g in cols:
                    xs = [r[metric] for r in runs.get((*g, d, m), {}).values()]
                    cells.append(ms(xs) + ("" if len(xs) >= PER_CELL else f" (n={len(xs)})"))
                out.append(f"| {m} | " + " | ".join(cells) + " |")
            avg = []
            for g in cols:
                ds_ = [(_avg(runs, (*g, d, m), metric) - _avg(runs, ("baseline", "default", d, m), metric))
                       for m in MODELS if len(runs.get((*g, d, m), {})) >= PER_CELL
                       and len(runs.get(("baseline", "default", d, m), {})) >= PER_CELL]
                avg.append(f"{sign(st.fmean(ds_))}" if ds_ else "—")
            out += ["| *avg Δ vs baseline* | " + " | ".join(avg) + " |", ""]
    # cosine vs MD within flag_feat
    rows = {(d, m): paired(runs, "flag_feat", d, m) for d in ("amazon_text", "yelpchi_text") for m in MODELS}
    comp = [r for r in rows.values() if r["complete"] and r["f1"] and r["auc"]]
    if comp:
        f1 = st.fmean(r["f1"]["d"] for r in comp); auc = st.fmean(r["auc"]["d"] for r in comp)
        w = lambda k, c: sum(r[k]["cls"] == c for r in comp)
        sig = lambda k, sgn: sum(1 for r in comp if r[k]["p"] is not None and r[k]["p"] < 0.05 and sgn * r[k]["d"] > 0)
        out += [f"**Cosine vs FLAG-MD within flag_feat** ({len(comp)} complete pairs): avg Δ F1 {sign(f1)} "
                f"(W/L/T {w('f1','win')}/{w('f1','loss')}/{w('f1','tie')}, sig+/sig− {sig('f1',1)}/{sig('f1',-1)}); "
                f"avg Δ AUC {sign(auc)} (W/L/T {w('auc','win')}/{w('auc','loss')}/{w('auc','tie')}, "
                f"sig+/sig− {sig('auc',1)}/{sig('auc',-1)}). Same rules as §9."]
        for d in ("amazon_text", "yelpchi_text"):
            cd = [r for (dd, _), r in rows.items() if dd == d and r["complete"] and r["f1"]]
            if cd:
                out.append(f"- {DS[d]}: avg Δ F1 {sign(st.fmean(r['f1']['d'] for r in cd))}, "
                           f"avg Δ AUC {sign(st.fmean(r['auc']['d'] for r in cd))} over {len(cd)} backbones.")
    return "\n".join(out)


def sec_text(runs):
    """Raw text only (text variant, 5 x 5) vs flag: what the LLM branch adds."""
    lines = ["Average over the 7 backbones (cells with 25/25 runs only; the count of such backbones is shown). "
             "*LLM adds* = flag − raw text, same sampler family except that `text` uses the variant's default "
             "neighbourhood (full 2-hop on Reddit/Instagram, random top-3 on Amazon/YelpChi) while flag uses "
             "cosine / FLAG-MD.", "",
             "| Dataset | metric | baseline | raw text only | flag · cosine | flag · FLAG-MD | LLM adds (cos) | LLM adds (MD) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    total_text = sum(min(len(runs.get(("text", "default", d, m), {})), PER_CELL) for d in DATASETS for m in MODELS)
    for d in DATASETS:
        for k, name in (("f1", "F1-macro"), ("auc", "AUC")):
            full = [m for m in MODELS if len(runs.get(("text", "default", d, m), {})) >= PER_CELL]
            if not full:
                lines.append(f"| {DS[d]} | {name} | — | not yet (0/7) | — | — | — | — |"); continue
            a = lambda g: st.fmean(_avg(runs, (*g, d, m), k) for m in full)
            b, t, fc, fm = a(("baseline", "default")), a(("text", "default")), a(("flag", "cosine")), a(("flag", "md_K2_matched"))
            lines.append(f"| {DS[d]} | {name} | {b:.3f} | {t:.3f} ({len(full)}/7) | {fc:.3f} | {fm:.3f} | "
                         f"{sign(fc - t)} | {sign(fm - t)} |")
    lines += ["", f"Raw-text runs complete: **{total_text}/700**."]
    return "\n".join(lines)


def sec_amazon_video(runs):
    """The self-built Amazon Video dataset: all main-study variants."""
    d = "amazon_video_text"
    out = []
    for metric, label in (("f1", "F1-macro"), ("auc", "AUC")):
        out += [f"**Amazon Video, {label}** (mean ± std, 25 runs; n shown when below 25)", "",
                "| Backbone | " + " | ".join(GNAME[g] for g in GROUPS) + " |", "|---|" + "---:|" * len(GROUPS)]
        for m in MODELS:
            cells = []
            for g in GROUPS:
                xs = [r[metric] for r in runs.get((*g, d, m), {}).values()]
                cells.append(ms(xs) + ("" if len(xs) >= PER_CELL else f" (n={len(xs)})"))
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        avg = []
        for g in GROUPS:
            ds_ = [(_avg(runs, (*g, d, m), metric) - _avg(runs, ("baseline", "default", d, m), metric))
                   for m in MODELS if len(runs.get((*g, d, m), {})) >= PER_CELL
                   and len(runs.get(("baseline", "default", d, m), {})) >= PER_CELL]
            avg.append(sign(st.fmean(ds_)) if ds_ else "—")
        out += ["| *avg Δ vs baseline* | " + " | ".join(avg) + " |", ""]
    for v in ("flag", "flag_finetuned"):
        rows = [paired(runs, v, d, m) for m in MODELS]
        comp = [r for r in rows if r["complete"] and r["f1"] and r["auc"]]
        if not comp:
            continue
        w = lambda k, c: sum(r[k]["cls"] == c for r in comp)
        sig = lambda k, sgn: sum(1 for r in comp if r[k]["p"] is not None and r[k]["p"] < 0.05 and sgn * r[k]["d"] > 0)
        out.append(f"- **Cosine vs FLAG-MD, {v}** ({len(comp)} backbones): avg Δ F1 "
                   f"{sign(st.fmean(r['f1']['d'] for r in comp))} (W/L/T {w('f1','win')}/{w('f1','loss')}/{w('f1','tie')}, "
                   f"sig+/sig− {sig('f1',1)}/{sig('f1',-1)}); avg Δ AUC {sign(st.fmean(r['auc']['d'] for r in comp))} "
                   f"(W/L/T {w('auc','win')}/{w('auc','loss')}/{w('auc','tie')}, sig+/sig− {sig('auc',1)}/{sig('auc',-1)}).")
    return "\n".join(out)


def _cell(runs, key, metric):
    xs = [r[metric] for r in runs.get(key, {}).values()]
    return ms(xs) + ("" if len(xs) >= PER_CELL else f" (n={len(xs)})")


def sec_followups(runs):
    """D-007 follow-ups: z-scored baseline (C), flag_feat on Amazon Video (B), raw text with FLAG samplers (A)."""
    out = []
    n = lambda g, dsets: sum(min(len(runs.get((*g, d, m), {})), PER_CELL) for d in dsets for m in MODELS)
    fd = ["amazon_text", "yelpchi_text", "amazon_video_text"]
    a_done = n(("text", "cosine"), DATASETS) + n(("text", "md_K2_matched"), DATASETS)
    b_done = n(("flag_feat", "cosine"), ["amazon_video_text"]) + n(("flag_feat", "md_K2_matched"), ["amazon_video_text"])
    c_done = n(("baseline_z", "default"), fd)
    out += [f"Progress: **A** {a_done}/1400 · **B** {b_done}/350 · **C** {c_done}/525.", "",
            "**C + B: scaling vs adding text** (F1-macro / AUC, mean ± std; n when below 25).", ""]
    cols = [("baseline", "default"), ("baseline_z", "default"), ("flag_feat", "cosine"), ("flag_feat", "md_K2_matched")]
    names = ["baseline (raw)", "baseline_z (z-scored)", "flag_feat · cosine", "flag_feat · FLAG-MD"]
    dsn = {**DS, "amazon_video_text": "Amazon Video"}
    for metric, label in (("f1", "F1-macro"), ("auc", "AUC")):
        for d in fd:
            out += [f"*{dsn[d]}, {label}*", "", "| Backbone | " + " | ".join(names) + " |", "|---|" + "---:|" * len(cols)]
            for m in MODELS:
                out.append(f"| {m} | " + " | ".join(_cell(runs, (*g, d, m), metric) for g in cols) + " |")
            avg = []
            for g in cols:
                ds_ = [(_avg(runs, (*g, d, m), metric) - _avg(runs, ("baseline", "default", d, m), metric))
                       for m in MODELS if len(runs.get((*g, d, m), {})) >= PER_CELL]
                avg.append(sign(st.fmean(ds_)) + ("" if len(ds_) == 7 else f" ({len(ds_)}/7)") if ds_ else "—")
            out += ["| *avg Δ vs raw baseline* | " + " | ".join(avg) + " |", ""]
    out += ["**A: what the LLM text adds with the same neighbourhoods** (average over backbones with 25/25 runs on both "
            "sides; count shown).", "",
            "| Dataset | metric | raw text · default nbhd | raw text · cosine | raw text · FLAG-MD | flag · cosine | flag · FLAG-MD | "
            "LLM adds (cos) | LLM adds (MD) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        for k, label in (("f1", "F1-macro"), ("auc", "AUC")):
            row = []
            for s_ in ("cosine", "md_K2_matched"):
                full = [m for m in MODELS if len(runs.get(("text", s_, d, m), {})) >= PER_CELL]
                row.append(full)
            fc, fm = row
            a = lambda g, ms_: st.fmean(_avg(runs, (*g, d, m), k) for m in ms_) if ms_ else None
            td = a(("text", "default"), MODELS)
            tc, tm = a(("text", "cosine"), fc), a(("text", "md_K2_matched"), fm)
            flc, flm = a(("flag", "cosine"), fc), a(("flag", "md_K2_matched"), fm)
            f = lambda x: "—" if x is None else f"{x:.3f}"
            out.append(f"| {DS[d]} | {label} | {f(td)} | {f(tc)} ({len(fc)}/7) | {f(tm)} ({len(fm)}/7) | {f(flc)} | {f(flm)} | "
                       f"{sign(flc - tc) if tc is not None else '—'} | {sign(flm - tm) if tm is not None else '—'} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    runs = load_runs()
    sections = {"progress": sec_progress(runs), "results_f1": sec_results(runs, "f1"),
                "results_auc": sec_results(runs, "auc"), "compare": sec_compare(runs),
                "vs_baseline": sec_vs_baseline(runs), "before_after": sec_before_after(runs),
                "coverage": sec_coverage(), "timing": sec_timing(runs),
                "feat": sec_feat(runs), "text": sec_text(runs), "amazon_video": sec_amazon_video(runs),
                "followups": sec_followups(runs)}
    if args.check:
        for k, v in sections.items():
            print(f"\n===== {k} =====\n{v}")
        return 0
    text = REPORT.read_text(encoding="utf-8")
    for k, v in sections.items():
        pat = re.compile(rf"<!-- AUTO:{k} BEGIN -->.*?<!-- AUTO:{k} END -->", re.S)
        if not pat.search(text):
            raise SystemExit(f"marker AUTO:{k} missing in {REPORT}")
        text = pat.sub(lambda _m, k=k, v=v: block(k, v), text)
    REPORT.write_text(text, encoding="utf-8")
    print(f"updated {REPORT.relative_to(ROOT)} ({', '.join(sections)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
