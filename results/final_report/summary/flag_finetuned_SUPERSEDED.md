# flag_finetuned (+FLAG*) - not in this (earlier) report

This report was built from `results/raw/` and `results/flag_md/` only, which hold no flag_finetuned runs.
flag_finetuned has since been run (7 backbones x 5 datasets x {cosine, FLAG-MD K=2}, 5 seeds x 5 inits,
175 runs per dataset/sampler, under `results/main/` and `results/main_gpu/`). Those results are in
`results/2026-10-02-flag-cosine-vs-md-main-run-report.md` (sections 8-10).
Per decision D-001 the LLM stays frozen, so every run carries `llm_finetuned=false`.
