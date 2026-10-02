#!/usr/bin/env python3
"""Starts server.py on 8799, checks /health, /api/status and the page, stops it. Exit 0 = pass."""
import json, subprocess, sys, time, urllib.request as u
p = subprocess.Popen([sys.executable, "server.py", "8799"])
get = lambda path: u.urlopen("http://127.0.0.1:8799" + path, timeout=30).read()
try:
    for _ in range(40):
        try: json.loads(get("/health")); break
        except OSError: time.sleep(.25)
    s = json.loads(get("/api/status"))
    for key in ("state", "done", "total", "groups", "open_cells", "gpus", "llm_caches", "failures",
                "sampling", "gpu_history"):
        assert key in s, f"/api/status missing {key}"
    assert s["state"] in ("running", "done", "failing", "idle"), s["state"]
    blocks = json.load(open("experiments.json"))
    want_groups = sum(len(b["groups"]) for b in blocks)
    want_total = sum(25 * len(b["groups"]) * len(b["datasets"]) * len(b.get("models", range(7))) for b in blocks)
    assert len(s["groups"]) == want_groups and s["total"] == want_total, (len(s["groups"]), s["total"], want_total)
    assert len(s["experiments"]) == len(blocks)
    cell = s["cells"][0]
    assert {"n", "f1_mean", "f1_std", "auc_mean", "auc_std"} <= set(cell["metrics"]), cell
    assert len(s["sampling"]) == 8, s["sampling"]
    assert b"<h1>" in get("/") and b"--bg" in get("/palette.css")
    print("OK")
finally:
    p.terminate(); p.wait()
