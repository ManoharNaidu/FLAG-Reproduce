#!/usr/bin/env python3
"""flag-dashboard (v4): read-only loopback service reporting FLAG experiment progress.

v4: the runs it tracks are defined in experiments.json (re-read every refresh), as
blocks of variant·sampler groups x datasets x backbones, so new experiments need no
code change.

v3 adds, all read-only: per-cell result metrics (n, mean/std of test F1-macro and
AUC) for the cosine vs MD comparison, sampling-cache status per dataset x sampler,
and ~10 minutes of in-memory GPU utilisation history for sparklines.

It never writes into the repo and never starts or stops experiment processes.
Everything it reads is cheap: result-file names (each JSON parsed once, then
cached), the tail of a few log files, cache manifests, /proc and nvidia-smi.
The snapshot is rebuilt at most every REFRESH seconds however often the page
polls, and the process lowers its own CPU priority at start-up.
"""
import collections, datetime, glob, json, os, re, statistics, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("FLAG_REPO") or os.path.abspath(os.path.join(D, "..", ".."))
REFRESH = float(os.environ.get("REFRESH", "20"))

DATASETS = ["reddit", "instagram", "amazon_text", "yelpchi_text"]
MODELS = ["bwgnn", "care_gnn", "dga_gnn", "gat", "gcn", "geniepath", "pmp"]
EXPERIMENTS_FILE = os.path.join(D, "experiments.json")


def _experiments():
    """Experiment blocks from experiments.json; falls back to the main study."""
    try:
        blocks = json.load(open(EXPERIMENTS_FILE))
    except (OSError, ValueError):
        blocks = [{"name": "Main study", "groups": [list(g) for g in GROUPS], "datasets": DATASETS}]
    for b in blocks:
        b.setdefault("models", MODELS); b.setdefault("note", "")
        b["groups"] = [tuple(g) for g in b["groups"]]
    return blocks


GROUPS = [("baseline", "default"), ("flag", "cosine"), ("flag", "md_K2_matched"),
          ("flag_finetuned", "cosine"), ("flag_finetuned", "md_K2_matched")]
STRATEGY_TO_SAMPLER = {"semantic": "cosine", "markov_diffusion": "md_K2_matched"}
PER_CELL = 25                       # 5 seeds x 5 inits
RUN_RE = re.compile(r"__s(\d+)i(\d+)__")
PROMPTS_RE = re.compile(r"Processed prompts:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+)")

_parsed = {}                        # path -> (status, training_time, error)
_lock = threading.Lock()
_snap = {"t": 0, "data": None}


def _read_result(path):
    if path not in _parsed:
        try:
            r = json.load(open(path))
            # Real wall-clock time of the run: file written at the end, `timestamp`
            # set at the start. `training_time` excludes flag_finetuned's extra
            # fine-tuning epochs, so it underestimates those runs 3-6x.
            try:
                wall = os.path.getmtime(path) - datetime.datetime.fromisoformat(r["timestamp"]).timestamp()
                wall = wall if wall > 0 else None
            except (KeyError, ValueError, TypeError):
                wall = None
            _parsed[path] = (r.get("status", "?"), r.get("training_time") or 0.0,
                             (r.get("error_message") or "")[:200],
                             r.get("test_f1_macro"), r.get("test_auc"), wall)
        except (OSError, ValueError):
            return None             # being written right now; retry next refresh
    return _parsed[path]


def _tail(path, n=6000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _elapsed(pid):
    """Seconds since process `pid` started, from /proc (None if unavailable)."""
    try:
        start = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19])
        return float(open("/proc/uptime").read().split()[0]) - start / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _processes():
    """Running train / LLM processes, from /proc (no subprocess spawned)."""
    train, llm, sampling = [], [], []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            args = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        args = [a.decode("utf-8", "replace") for a in args if a]
        if "scripts.train.run" in args:
            def opt(*names):
                for name in names:
                    if name in args and args.index(name) + 1 < len(args):
                        return args[args.index(name) + 1]
                return ""
            strategy = opt("--sampling-strategy")
            train.append({
                "dataset": opt("--dataset", "--datasets"), "model": opt("--model", "--models"),
                "variant": opt("--variant", "--variants"),
                "sampler": STRATEGY_TO_SAMPLER.get(strategy, "default" if not strategy else strategy),
                "device": "gpu" if opt("--device").startswith("cuda") else "cpu",
                "elapsed": _elapsed(pid),
            })
        elif "scripts.llm.generate_text" in args or "scripts.preprocess.encode_llm_text" in args:
            llm.append(" ".join(args[-12:]))
        elif "scripts.preprocess.sample_subgraphs" in args:
            sampling.append(" ".join(args))
    return train, llm, sampling


def _gpus():
    q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 8:
            continue
        num = lambda x: float(x) if x.replace(".", "", 1).isdigit() else None
        gpus.append({"index": int(f[0]), "name": f[1], "util": num(f[2]), "mem_used": num(f[3]),
                     "mem_total": num(f[4]), "temp": num(f[5]), "power": num(f[6]), "power_limit": num(f[7])})
    return gpus


def _llm_caches():
    rows = []
    for m in sorted(glob.glob(os.path.join(REPO, "cache/llm/*.manifest.json"))):
        try:
            d = json.load(open(m))
        except (OSError, ValueError):
            continue
        if d.get("llm_config", {}).get("engine") != "vllm":
            continue                # only the D-005 production caches
        s = d.get("stats", {})
        rows.append({"dataset": d.get("dataset"), "kind": d.get("kind"),
                     "sampler": STRATEGY_TO_SAMPLER.get(d.get("sampling_config", {}).get("strategy"), "?"),
                     "coverage": s.get("node_coverage"), "success": s.get("subgraph_success_rate"),
                     "subgraphs": s.get("total_subgraphs"), "generated_at": d.get("generated_at")})
    return rows


def _llm_live():
    out = []
    for log in sorted(glob.glob(os.path.join(REPO, "logs/llm/vllm_*_gpu*.log"))):
        if time.time() - os.path.getmtime(log) > 300:
            continue                # not written in 5 min: not live
        text = _tail(log)
        job = re.findall(r"^([A-Z_]{4,})\s+kind=(\w+)", text, re.M)
        prog = PROMPTS_RE.findall(text.replace("\r", "\n"))
        if prog:
            pct, done, total, elapsed, left = prog[-1]
            out.append({"log": os.path.basename(log), "job": " ".join(job[-1]).lower() if job else "",
                        "percent": int(pct), "done": int(done), "total": int(total), "left": left})
    return out


def _queued_cells():
    """Cells with runs waiting in the parallel queue (logs/main/parallel/jobs.txt,
    one 'wall variant sampler dataset model seed init' line per run)."""
    out = set()
    try:
        for path in glob.glob(os.path.join(REPO, "logs/main/parallel/jobs*.txt")):
            for line in open(path):
                f = line.split()
                if len(f) == 7:
                    out.add((f[1], f[2], f[3], f[4]))
    except OSError:
        pass
    return out


def _failures_from_logs():
    """FAIL lines from the matrix logs as (variant, sampler, dataset, model) cells,
    plus tracebacks in the GPU run logs. build() drops a FAIL whose cell has since
    completed or is running again (e.g. a job deliberately moved to the GPUs)."""
    cells, other = [], []
    for variant in ("baseline", "flag", "flag_finetuned"):
        for line in _tail(os.path.join(REPO, "logs/main", f"{variant}_matrix.log"), 200000).splitlines():
            f = line.split()
            if len(f) == 4 and f[0] == "FAIL":
                cells.append((variant, f[3], f[1], f[2]))
    for path in glob.glob(os.path.join(REPO, "logs/main/gpu/*.log")):
        if "Traceback" in _tail(path, 20000):
            other.append(f"Traceback in logs/main/gpu/{os.path.basename(path)}")
    return cells, other


def _summarise(metrics):
    """n, mean and sample std (n-1) of test F1-macro and AUC over unique runs."""
    f1 = [v[0] for v in metrics.values()]; auc = [v[1] for v in metrics.values()]
    sd = lambda xs: round(statistics.stdev(xs), 5) if len(xs) > 1 else None
    mean = lambda xs: round(statistics.fmean(xs), 5) if xs else None
    return {"n": len(f1), "f1_mean": mean(f1), "f1_std": sd(f1), "auc_mean": mean(auc), "auc_std": sd(auc)}


# Sampling caches per dataset x sampler. top_k mirrors the registry's per-dataset
# default (10, or 3 on the dense Amazon/YelpChi graphs; decision D-005).
TOP_K = {"reddit": 10, "instagram": 10, "amazon_text": 3, "yelpchi_text": 3}
SAMPLING_KEYS = {"cosine": "semantic_h2_k{k}_t0_perhop",
                 "md_K2_matched": "markov_diffusion_h2_k{k}_t0_perhop_K2_matched_cosine"}


def _sampling(sampling_procs):
    rows = []
    for sampler, pattern in SAMPLING_KEYS.items():
        for ds in DATASETS:
            stem = os.path.join(REPO, "cache/sampling", f"{ds}__{pattern.format(k=TOP_K.get(ds, 3))}")
            if os.path.exists(stem + ".pt") and os.path.exists(stem + ".json"):
                status = "done"
            elif any(ds in a for a in sampling_procs):
                status = "running"
            else:
                status = "missing"
            rows.append({"dataset": ds, "sampler": sampler, "status": status})
    return rows


# GPU utilisation history, sampled by one background thread every HISTORY_EVERY s.
HISTORY_EVERY, HISTORY_POINTS = 20, 30          # ~10 minutes
_gpu_hist = collections.deque(maxlen=HISTORY_POINTS)


def _gpu_sampler():
    while True:
        try:
            g = _gpus()
            if g:
                _gpu_hist.append({"t": round(time.time()), "util": {x["index"]: x["util"] for x in g}})
        except Exception:
            pass
        time.sleep(HISTORY_EVERY)


def build():
    train_procs, llm_procs, sampling_procs = _processes()
    running, elapsed = {}, {}
    for p in train_procs:
        key = (p["variant"], p["sampler"], p["dataset"], p["model"])
        running.setdefault(key, []).append(p["device"])
        if p["elapsed"]:
            elapsed[key] = max(elapsed.get(key, 0), p["elapsed"])

    queued = _queued_cells()
    blocks = _experiments()
    cells, failures, seen = [], [], set()
    for block in blocks:
      for variant, sampler in block["groups"]:
        for ds in block["datasets"]:
            for model in block["models"]:
                if (variant, sampler, ds, model) in seen:
                    continue                    # a cell listed in two blocks counts once
                seen.add((variant, sampler, ds, model))
                runs, times, metrics, walls = set(), [], {}, []
                for root in ("results/main", "results/main_gpu"):
                    for f in glob.glob(os.path.join(REPO, root, variant, sampler, f"{ds}__{model}__*.json")):
                        m = RUN_RE.search(f); rec = _read_result(f)
                        if not m or rec is None:
                            continue
                        status, ttime, err, f1, auc, wall = rec
                        if status != "completed":
                            failures.append({"cell": f"{variant}/{sampler}/{ds}/{model}",
                                             "run": f"s{m.group(1)}i{m.group(2)}", "error": err or status})
                            continue
                        runs.add(m.groups()); times.append(ttime)
                        if wall:
                            walls.append(wall)
                        if f1 is not None and auc is not None:
                            metrics[m.groups()] = (f1, auc)
                devs = running.get((variant, sampler, ds, model), [])
                n = min(len(runs), PER_CELL)
                med = statistics.median(times) if times else None
                wall_med = statistics.median(walls) if walls else None
                # Per-run duration: measured wall-clock; else training_time; else, for a cell
                # with no finished run, how long its oldest process has been running (a
                # lower bound, flagged provisional).
                provisional = False
                per_run = wall_med or med
                if per_run is None and elapsed.get((variant, sampler, ds, model)):
                    per_run, provisional = elapsed[(variant, sampler, ds, model)], True
                eta = None
                if n < PER_CELL and per_run is not None:
                    if (variant, sampler, ds, model) in queued:
                        eta = per_run      # parallel queue: each remaining run gets its own process
                    else:
                        eta = (PER_CELL - n) * per_run / max(len(devs), 1)
                cells.append({"experiment": block["name"],
                              "variant": variant, "sampler": sampler, "dataset": ds, "model": model,
                              "done": n, "running": len(devs), "device": devs[0] if devs else "",
                              "median_s": round(med) if med else None, "wall_s": round(wall_med) if wall_med else None,
                              "eta_s": round(eta) if eta else None, "provisional": provisional,
                              "per_run_s": round(per_run) if per_run else None,
                              "metrics": _summarise(metrics)})

    groups, experiments = [], []
    for block in blocks:
        bcells = [c for c in cells if c["experiment"] == block["name"]]
        for variant, sampler in block["groups"]:
            sub = [c for c in bcells if c["variant"] == variant and c["sampler"] == sampler]
            groups.append({"experiment": block["name"], "variant": variant, "sampler": sampler,
                           "done": sum(c["done"] for c in sub), "total": PER_CELL * len(sub)})
        bdone, btotal = sum(c["done"] for c in bcells), PER_CELL * len(bcells)
        experiments.append({"name": block["name"], "note": block["note"], "done": bdone, "total": btotal,
                            "running": sum(c["running"] for c in bcells),
                            "datasets": block["datasets"], "groups": [list(g) for g in block["groups"]]})
    open_cells = [c for c in cells if c["done"] < PER_CELL]
    # Overall ETA: remaining work spread over the processes running now, but never
    # less than the longest single remaining run (runs cannot be split).
    per_run = [c["per_run_s"] for c in open_cells if c["per_run_s"]]
    work = sum((PER_CELL - c["done"]) * c["per_run_s"] for c in open_cells if c["per_run_s"])
    slots = max(len(train_procs), 1)
    etas = [max(max(per_run), work / slots)] if per_run else []
    unknown_eta = any(c["eta_s"] is None or c["provisional"] for c in open_cells)
    fail_cells, log_failures = _failures_from_logs()
    by_key = {(c["variant"], c["sampler"], c["dataset"], c["model"]): c for c in cells}
    superseded = 0
    for key in dict.fromkeys(fail_cells):
        c = by_key.get(key)
        if c and (c["done"] >= PER_CELL or c["running"] or key in queued):
            superseded += 1
        else:
            log_failures.append("matrix job failed: " + "/".join(key))
    done = sum(g["done"] for g in groups); total = sum(g["total"] for g in groups)
    if failures or log_failures:
        state = "failing"
    elif done >= total:
        state = "done"
    elif train_procs or llm_procs:
        state = "running"
    else:
        state = "idle"
    return {
        "updated": time.time(), "state": state, "done": done, "total": total,
        "eta_s": max(etas) if etas else None, "eta_partial": unknown_eta,
        "groups": groups, "experiments": experiments, "open_cells": sorted(open_cells, key=lambda c: -(c["eta_s"] or 1e9)),
        "cells": cells, "failures": failures[:50], "log_failures": log_failures[:50], "superseded": superseded,
        "procs": {"train_cpu": sum(p["device"] == "cpu" for p in train_procs),
                  "train_gpu": sum(p["device"] == "gpu" for p in train_procs), "llm": len(llm_procs)},
        "gpus": _gpus(), "llm_caches": _llm_caches(), "llm_live": _llm_live(),
        "sampling": _sampling(sampling_procs), "gpu_history": list(_gpu_hist),
    }


def snapshot():
    with _lock:
        if _snap["data"] is None or time.time() - _snap["t"] >= REFRESH:
            try:
                _snap["data"] = build(); _snap["error"] = None
            except Exception as e:      # keep serving the last good snapshot
                _snap["error"] = f"{type(e).__name__}: {e}"
            _snap["t"] = time.time()
        return {**(_snap["data"] or {}), "error": _snap.get("error")}


def css():
    p = json.load(open(os.path.join(D, "palette.json")))
    v = lambda c: ";".join(f"--{k}:{x}" for k, x in c.items())
    return f":root{{{v(p['light'])}}}@media(prefers-color-scheme:dark){{:root{{{v(p['dark'])}}}}}"


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ct="application/json"):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/": self.send(200, open(os.path.join(D, "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/palette.css": self.send(200, css(), "text/css")
        elif self.path == "/health":
            vf = os.path.join(D, "VERSION")
            self.send(200, {"ok": True, "repo": REPO, "version": open(vf).read().strip() if os.path.exists(vf) else "dev"})
        elif self.path == "/api/status": self.send(200, snapshot())
        else: self.send(404, {"error": "not found"})

    def log_message(self, *a): pass


if __name__ == "__main__":
    try:
        os.nice(19)                     # never compete with the experiments for CPU
    except OSError:
        pass
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8780
    threading.Thread(target=_gpu_sampler, daemon=True).start()
    HTTPServer(("127.0.0.1", port), H).serve_forever()   # loopback only, single-threaded
