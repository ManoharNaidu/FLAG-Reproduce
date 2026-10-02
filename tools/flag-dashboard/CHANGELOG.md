## 0.4.0 (2026-10-02)
Experiments: tracked runs are defined in experiments.json (blocks of variant·sampler groups x datasets x
backbones, re-read every refresh), so new experiments need no code change. Page gains an Experiments strip,
header totals across all blocks, flag_feat in the cosine vs MD comparison and filters; the pipeline is scoped
to the main study. Server: `experiments` summary, `experiment` tag per cell, all logs/main/parallel/jobs*.txt
queues recognised. Previous files kept as 2026-10-02-server-v3.1-before-v4.py and
2026-10-02-index-v3-before-v4.html.

## 0.3.1 (2026-10-01)
ETA fix (server.py): per-run duration is measured wall-clock (result timestamp to file write) instead of
training_time, which omits flag_finetuned's fine-tuning epochs (3-6x too low). Cells with no finished run use
their oldest process's elapsed time as a provisional lower bound. Overall ETA = max(longest single remaining
run, remaining work / running processes); queued cells in the parallel queue need about one run once started.
Matrix jobs stopped and requeued in logs/main/parallel/jobs.txt count as superseded, not failed.
New cell fields: wall_s, per_run_s, provisional. Previous server kept as 2026-10-01-server-v3-before-wall-eta.py.

## 0.3.0 (2026-10-01)
Pipeline view: five stage cards (sampling, LLM text, fine-tuning, GNN training, results) with cosine and
FLAG-MD lanes, status, blockers and a "limiting" marker. New "Results: cosine vs MD" comparison: paired
mean ± std (25 runs) of test F1-macro and AUC, Δ = FLAG-MD − cosine coloured win/loss/noise, partial cells
marked and excluded, summary by variant and dataset, filters, baseline reference rows, Instagram coverage
warning. Stage 3 marked "not tracked" (LLM frozen in this reproduction, decision D-001). Unfinished-cells
table gains a sampler filter; GPU rows gain a utilisation sparkline.
server.py (read-only additions): per-cell metrics (n, mean, sample std), sampling-cache status, ~10 min
in-memory GPU utilisation history. Previous files kept as 2026-10-01-server-v2-before-pipeline.py and
2026-10-01-index-v2-before-pipeline.html.

## 0.2.0 (2026-10-01)
Redesign (layout/visuals only, same data and API): overall progress bar; critical-path row and header note;
"Xh YYm" times; blue/green/red/amber colour roles; two columns at >=1200px; unfinished cells grouped by
dataset with filter chips, sortable columns, GPU/CPU/Queued badges; GPU util + memory bars; coverage heatmap
with legend; empty sections collapse to one line. Previous page kept as 2026-10-01-index-before-redesign.html.

## 0.1.0 (2026-10-01)
First version: read-only FLAG progress dashboard (runs by variant, GPUs, unfinished cells with ETA, failures, LLM cache coverage). Loopback service on 8780, nice 19, 20 s snapshot cache; exposed through the instance portal as "FLAG Dashboard" (external port 10100, token auth).

