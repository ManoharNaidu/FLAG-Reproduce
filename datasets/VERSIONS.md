# Dataset Versions

Frozen record of exactly which artefacts these datasets were built from
(brief section 28). Checksums for the downloaded sources live in
`datasets/manifests/*.json`; checksums for the built payloads live in
`datasets/processed/<name>/manifest.json`.

## YelpChi

| field | value |
|---|---|
| graph source | CARE-GNN `YelpChi.mat` |
| graph sha256 | `fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42` |
| text source | Mukherjee et al. ICWSM 2013 corpus, via mirror `github.com/zyni2001/Anomaly-detection-LLM` |
| text files | `metadata.txt` (67,395 rows), `raw_text.txt` (5,854 hotel), `output_review_yelpResData_NRYRcleaned.txt` (61,541 restaurant) |
| node count | 45,954 (subset of the 67,395-review corpus: products with >800 reviews dropped) |
| feature dim | 32 |
| relations | rur, rtr, rsr |
| node ordering | user first-appearance, **proven** by adjacency reconstruction |
| mapping method | adjacency proof (R-U-R 98,630 + R-S-R 6,805,486, 0 mismatches) |
| text join | positional within the concatenated hotel+restaurant corpus; statistical support only |
| known gap | R-T-R not reproducible (date snapshot differs) |

## Amazon

| field | value |
|---|---|
| graph source | CARE-GNN `Amazon.mat` |
| graph sha256 | `4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63` |
| text source | McAuley 2014 `reviews_Musical_Instruments.json.gz` — the **full**, non-5-core dump (500,176 reviews / 339,231 reviewers) |
| wrong versions ruled out | 2014 5-core (1,429 reviewers), 2018 5-core (27,530 reviewers) |
| node count | 11,944 = 3,305 sampled unlabelled + 8,639 labelled |
| feature dim | 25 (literature's "24" unverified; DGL does not slice) |
| relations | upu, usu, uvu |
| labelling rule | `votes >= 20`, ratio `> 0.8` benign / `< 0.2` fraud (CARE-GNN's thresholds; the >=20 filter originates with Zhang et al. SIGIR 2020) |
| node ordering | `{**sampled, **labeled}` -> unlabelled prefix `[0, 3305)` |
| mapping method | generator order + induced U-P-U proof (294,764 edges, 0 mismatches) |
| known gap | the 3,305 unlabelled nodes are UNRESOLVED |

## Provenance chains

- **YelpChi**: Mukherjee et al. ICWSM 2013 (corpus) -> Rayana & Akoglu KDD 2015
  (32 features) -> CARE-GNN CIKM 2020 (graph + 45,954 subset).
- **Amazon**: McAuley & Leskovec 2013 (corpus) -> REV2 WSDM 2018 (helpful-votes
  heuristic) -> Zhang et al. SIGIR 2020 (25 features, >=20-vote filter) ->
  CARE-GNN CIKM 2020 (Musical Instruments, 0.8/0.2, U-P-U/U-S-U/U-V-U).

CARE-GNN constructed the graphs but originated neither the features nor the
labelling heuristics.
