# Loop 1 / Task E — Independent provenance check (YelpChi + Amazon)

Written incrementally. Structure: **CLAIM → SOURCE → AGREE / CONFLICT / UNKNOWN**.
No conflict is resolved by picking a winner; both sides are recorded.

Status legend:
- `AGREE` — every source consulted says the same thing.
- `CONFLICT` — sources consulted disagree; both are recorded verbatim.
- `UNKNOWN` — not established by anything consulted. Not guessed.

---

## Q1 — Does the FLAG paper report ANY YelpChi / Amazon result?

### C1.1 — FLAG's result tables cover exactly three datasets: Reddit, Instagram, Huabei.

**Source (local, transcribed from the paper):**
`research/reported_results.csv` — the full transcription of FLAG's published
numbers. Distinct `(source_table, dataset)` pairs in the file:

```
Table 3 | huabei_industrial   (bwgnn, care_gnn, dga_gnn, flag, flag_star, gat, gcn, geniepath, pmp)
Table 4 | instagram           (bwgnn, care_gnn, dga_gnn, gat, gcn, geniepath, pmp)
Table 4 | reddit              (bwgnn, care_gnn, dga_gnn, gat, gcn, geniepath, pmp)
Table 5 | instagram           (bwgnn, care_gnn, gat, gcn, geniepath)   [ablation]
Table 5 | reddit              (bwgnn, care_gnn, gat, gcn, geniepath)   [ablation]
```

No row mentions `yelp`, `yelpchi`, `amazon`, `t-finance`, `t-social` or `elliptic`
as a *dataset*. (`care_gnn` appears only as a **baseline model name**, which is
the likely source of confusion — CARE-GNN the *method* is a FLAG baseline, but
CARE-GNN's *datasets* are not used.)

**Source (local, structural):** `research/paper_notes.md:413-421`
> "- **Table 3** — industrial Huabei: 7 baselines (all `+text`) + FLAG + FLAG\*,
> - **Table 4** — Reddit and Instagram, 7 models x 4 variants x 2 metrics,
> - **Table 5** — ablation over SS / LLM / SG. **Only 5 backbones** (GCN, GAT, ...)"

**Verdict: AGREE.** Tables 1-2 are prompt templates; Tables 3-5 are results over
{Huabei, Reddit, Instagram} only. **FLAG reports zero YelpChi/Amazon numbers,
under any variant (`baseline`, `+text`, `+FLAG`, `+FLAG*`) and in any ablation
cell.**

### C1.2 — The paper's own sentence explaining *why* YelpChi/Amazon are excluded.

**Source:** FLAG (KDD '25), DOI `10.1145/3711896.3737220`, quoted in
`research/paper_notes.md:33-39` and independently again in
`research/dataset_notes.md:322-326`. Verbatim:

> "for the public fraud detection datasets, we note that most of them, such as
> Yelp-Fraud [8], Amazon-Fraud [8], T-Finance [36] and T-Social [36], lack
> textual information... To overcome this limitation, we construct a dataset
> tailored to fraud detection by utilizing existing social network datasets that
> contain textual data. Specifically, we use two social network datasets:
> Reddit [25] and Instagram [25]."

Location in the paper: the **Datasets** paragraph of the experimental-setup
section (the paragraph that introduces Reddit/Instagram). Citation `[8]` is
CARE-GNN (Dou et al., CIKM 2020) — i.e. FLAG points at exactly the `.mat`
release we hold. Precise section/page number: **UNKNOWN** from the local notes
(the notes quote the sentence but do not record §/page).

**Verdict: AGREE** (two independent local transcriptions agree verbatim).

### C1.3 — Consequence (stated, not inferred beyond the two claims above)

A text-augmented YelpChi or Amazon run **cannot** be compared against any
published FLAG number, because no such number exists. Any such run is a
**novel experiment**, not a reproduction. This matches the repo's own already-
recorded policy at `research/dataset_notes.md:328-345` and
`research/dataset_integration_audit.md:129-145`, which hard-blocks
`variant=flag` / `variant=flag_finetuned` for these datasets via
`DATASET_REGISTRY` (`registry.py:284-320`, `has_native_text=False`).

**Verdict: AGREE** — local policy and the paper's own sentence are consistent.

---

## Q2 — The Amazon 24-vs-25 feature conflict

### C2.1 — Every primary/secondary source consulted says **25**. No source saying 24 was found.

| Source | URL | States |
|---|---|---|
| CARE-GNN (Dou et al., CIKM 2020) §4.1.1 | https://ar5iv.labs.arxiv.org/html/2008.08692 | "25 handcrafted features from (Zhang et al. 2020)" |
| DGL source `python/dgl/data/fraud.py` docstring | https://raw.githubusercontent.com/dmlc/dgl/master/python/dgl/data/fraud.py | Amazon "25 handcrafted features"; Yelp "32 handcrafted features" |
| DGL docs `FraudAmazonDataset` (2.5) | https://www.dgl.ai/dgl_docs/generated/dgl.data.FraudAmazonDataset.html | "Node features: 25 handcrafted features" |
| GADBench (NeurIPS'23 D&B) Table 3 | https://arxiv.org/html/2306.12251v1 | `Amazon \| 11,944 \| 4,398,392 \| 25 \| 9.5%` and `YelpChi \| 45,954 \| 3,846,979 \| 32 \| 14.5%` |
| SEFraud (arXiv:2406.11389) stats table | https://arxiv.org/pdf/2406.11389 | Amazon: 11,944 nodes, **25** features, 4,398,392 edges, 3 edge types, 2 classes |

**Our local file: 25.** → **AGREE with every published source consulted.**

### C2.2 — Where "24" plausibly comes from — CANDIDATE ONLY, not established

Targeted searches for a published "24 features" claim about *this* benchmark
returned **nothing**. One lead surfaced: **SEFraud (arXiv:2406.11389)** puts
Amazon (25 features) in the same statistics table as its private **ICBC**
financial-fraud dataset, which has **24** features. A reader skimming that table
could carry "24" across rows. This is a hypothesis, not a citation.

**Status: UNKNOWN / UNSUBSTANTIATED.** The premise "different sources report 24
or 25" is **not confirmed by this check**. If the 24 claim has a real source, it
was not located, and it is *not* DGL, *not* CARE-GNN, *not* GADBench.
**Do not treat 24-vs-25 as a live version conflict** on the evidence collected.
Marked UNKNOWN rather than resolved in either direction.

### C2.3 — DGL does NOT slice the feature matrix.

Verbatim from DGL `FraudDataset.process` (master):

```python
data = io.loadmat(file_path)
node_features = data["features"].todense()
node_labels = data["label"].squeeze()
```

No column slicing, no dropping. Whatever width the `.mat` has (25), DGL passes
through. **So "DGL's loader slices to 24" is FALSE.** The 25 in our file and the
25 DGL documents are the *same* 25.

### C2.4 — The underlying enumeration is genuinely ambiguous (a real, separate issue)

Zhang et al. (SIGIR 2020, arXiv:2005.10150) Table 1 — the origin of the feature
set — is a **bulleted prose list**, not 25 numbered rows. Several bullets expand
to several columns each ("Number and ratio of each rating level given by a user"
is 10 columns; "Median, min, and max number of helpful and unhelpful votes" is 6).
See sibling evidence `research/_evidence/loop1_b_amazon_raw.md:339-360` for the
verbatim list.

**This** — not a loader — is the credible mechanism by which a careful reader
could arrive at a count other than 25: the source never enumerates 25 atomic
features, it describes feature *families*. **CONFLICT-BY-AMBIGUITY, unresolved.**
Anyone claiming a specific per-column semantic mapping for `Amazon.mat` must
prove it by recomputation, not by reading Table 1.

---

## Q3 — Is DGL's FraudYelp / FraudAmazon tensor-identical to CARE-GNN's `.mat`?

Primary source, DGL master:
https://raw.githubusercontent.com/dmlc/dgl/master/python/dgl/data/fraud.py

### C3.1 — Features, labels, edges: pass-through, NO transformation. AGREE / IDENTICAL.

`process()` verbatim:

```python
file_path = os.path.join(self.raw_path, self.file_names[self.name])
data = io.loadmat(file_path)
node_features = data["features"].todense()
# remove additional dimension of length 1 in raw .mat file
node_labels = data["label"].squeeze()

graph_data = {}
for relation in self.relations[self.name]:
    adj = data[relation].tocoo()
    row, col = adj.row, adj.col
    graph_data[(self.node_name[self.name], relation, self.node_name[self.name])] = (row, col)
g = heterograph(graph_data)

g.ndata["feature"] = F.tensor(node_features, dtype=F.data_type_dict["float32"])
g.ndata["label"]   = F.tensor(node_labels,  dtype=F.data_type_dict["int64"])
```

Explicitly **absent**: row-normalisation, self-loops, `add_reverse_edges`,
symmetrisation, any node permutation / `reorder_graph`, any feature scaling.
`.todense()` plus a `float32` cast is the only change; `.tocoo()` preserves the
`.mat` sparsity pattern exactly. **Node ordering is the `.mat` row order,
untouched** — DGL node *i* is `.mat` row *i*.

### C3.2 — Edge counts: DGL / our counts are DOUBLE CARE-GNN's published table. RECONCILED.

| Relation | CARE-GNN paper table | DGL docs / our `.mat` read | Ratio |
|---|---|---|---|
| Yelp R-U-R | 49,315 | 98,630 | 2x |
| Yelp R-T-R | 573,616 | 1,147,232 | 2x |
| Yelp R-S-R | 3,402,743 | 6,805,486 | 2x |
| Amazon U-P-U | 175,608 | 351,216 | 2x |
| Amazon U-S-U | 3,566,479 | 7,132,958 | 2x |
| Amazon U-V-U | 1,036,737 | 2,073,474 | 2x |

CARE-GNN (https://ar5iv.labs.arxiv.org/html/2008.08692) reports **undirected
edges**; the `.mat` adjacency is **symmetric**, so a COO read yields exactly 2x
as directed entries. Our locally measured numbers match DGL's documented numbers
**exactly**, in all six relations.
**Not a conflict — a units difference. Both conventions recorded.**

Separate note: GADBench reports Amazon 4,398,392 and YelpChi 3,846,979 total
edges, which is **less** than the sum of the three relations (4,778,824 and
4,025,674 undirected respectively) — GADBench evidently **de-duplicates across
relations** when homogenising. A third convention. Recorded, not resolved.

### C3.3 — Node / class counts: AGREE, with one arithmetic caveat on Amazon.

- Yelp: 45,954 nodes / 32 feats / 39,277 benign / 6,677 spam — our read == DGL
  docs == CARE-GNN. **AGREE.**
- Amazon: 11,944 nodes / 25 feats. DGL docs state "Positive (fraudulent): 821,
  Negative (benign): 7,818, Unlabeled: 3,305". Our read said "11,123 / 821".
  `7,818 + 3,305 = 11,123` — our benign count **absorbs the 3,305 unlabeled
  nodes**, which carry `label == 0` in the `.mat`. Same tensor, different
  bookkeeping. **AGREE after reconciliation — but the local figure must be
  relabelled "label==0 (incl. unlabeled)", not "benign".**

### C3.4 — CONFLICT: Amazon fraud ratio 9.5% vs 6.9%

- CARE-GNN table and GADBench Table 3 both print Amazon fraud rate **9.5%**
  = 821 / 8,639 (**labeled nodes only**).
- `research/dataset_integration_audit.md:110` uses **6.9%** = 821 / 11,944
  (**all nodes**).

Both are arithmetically correct over different denominators. **CONFLICT in
reported convention — surfaced, not resolved.** Any table we publish must state
its denominator explicitly or it will silently disagree with the literature.

### C3.5 — DGL DOES add something the `.mat` does not have: its own random splits.

`_random_split` verbatim, with `__init__` defaults `random_seed=717,
train_size=0.7, val_size=0.1`:

```python
N = x.shape[0]
index = np.arange(N)
if self.name == "amazon":
    # 0-3304 are unlabeled nodes
    index = np.arange(3305, N)

index = np.random.RandomState(seed).permutation(index)
train_idx = index[: int(train_size * len(index))]
val_idx   = index[len(index) - int(val_size * len(index)) :]
test_idx  = index[int(train_size * len(index)) : len(index) - int(val_size * len(index))]
...
self.graph.ndata["train_mask"] = F.tensor(train_mask)
self.graph.ndata["val_mask"]   = F.tensor(val_mask)
self.graph.ndata["test_mask"]  = F.tensor(test_mask)
```

Two consequences:

1. DGL's default is a **70/10/20** split at `seed=717`. CARE-GNN's own code uses
   `train_test_split(test_size=0.60, random_state=2)` (i.e. 40% train), which the
   repo's native builder reproduces
   (`research/dataset_integration_audit.md:112-118`). **Different protocols over
   the same graph.** Numbers from a DGL-loaded run and a CARE-GNN-loaded run are
   **not comparable** even though the tensors are identical. Most published
   YelpChi/Amazon numbers must therefore be read together with which split they
   used — **which is frequently not stated. UNKNOWN per-paper.**
2. DGL **does** hard-code the Amazon unlabeled range `[0, 3305)` and excludes it
   from all three masks — independently confirming the local observation about
   indices `[0,3305)`. **AGREE.**

### C3.6 — Verdict on Q3

**Tensors: identical** (same node order, same feature matrix, same edges — DGL
applies no transformation). **Protocol: not identical** (DGL injects a 70/10/20
seed-717 split CARE-GNN never published). "Same dataset, different loader" is
safe at the tensor level here and **unsafe at the split level**.

---

## Q4 — Has anyone PUBLISHED a verified mapping from `.mat` row index back to original review/reviewer IDs?

### C4.0 — Answer, stated plainly

**No.** Across every source consulted, **no peer-reviewed publication and no
official dataset release ships, or claims to have verified, a row-index →
review-ID (YelpChi) or row-index → reviewerID (Amazon) mapping.** The only
artifact found that *asserts* such a mapping is an unpublished GitHub repository
that **assumes positional order** and never checks it (C4.3).

**We must not claim an EXACT mapping on the strength of anything found here.**

### C4.1 — The `.mat` files carry no identifier column at all.

DGL's loader reads exactly three key families and nothing else
(https://raw.githubusercontent.com/dmlc/dgl/master/python/dgl/data/fraud.py):
`data["features"]`, `data["label"]`, and the relation adjacencies
(`net_rur`/`net_rtr`/`net_rsr`, `net_upu`/`net_usu`/`net_uvu`). There is no
`review_id`, `reviewerID`, `user_id` or `date` array to join on.
**AGREE** with the local observation in
`research/dataset_integration_audit.md:32` ("no reviewer id, no text").

### C4.2 — The benchmark's own author confirms text existed, but never released the link.

Source: CARE-GNN issue #2, reply by repo owner **YingtongDou** (the CARE-GNN
first author), 2020-09-14, retrieved via GitHub API
(https://github.com/YingtongDou/CARE-GNN/issues/2#issuecomment-692061862).
Verbatim:

> "We use handcrafted features for both Yelp and Amazon datasets. The Yelp
> feature comes from Table 2 of the following paper: Shebuti Rayana, Leman
> Akoglu. Collective Opinion Spam Detection: Bridging Review Networks and
> Metadata. KDD 2015. We concatenate 15 review behavior&text features, 9 user
> behavior features, and 8 product behavior features of a review as a
> 32-dimension feature vector.
> We have verified that text embeddings are almost useless for the Yelp spam
> review classification task, but they might work on your own dataset, you can
> have a try."

Three things follow, all load-bearing:

1. **15 + 9 + 8 = 32.** YelpChi's feature width is fully accounted for.
   **AGREE** with our local read of 32.
2. The authors **did** hold the review text (they computed text embeddings from
   it) — so text-node correspondence existed *in their pipeline*. They never
   published it.
3. Follow-up in the same thread, verbatim: *"The python code for computing
   behavior features is
   [yelpFeatureExtraction.py](https://github.com/YingtongDou/Nash-Detect/blob/master/Utils/yelpFeatureExtraction.py).
   You can [contact](http://odds.cs.stonybrook.edu/yelpchi-dataset/) the Yelp
   dataset author for the Matlab code to compute text features."* — i.e. the
   text-feature code was **never released**; it is email-gated with a third
   party. A later commenter (`haophancs`) asks the same question for Amazon's 25
   dimensions; **no answer appears in the thread. UNKNOWN.**

### C4.3 — The single artifact that asserts a mapping: `zyni2001/Anomaly-detection-LLM` — category **(b), "assumes row order"**

- Repo: https://github.com/zyni2001/Anomaly-detection-LLM (created 2023-10-25,
  7 stars, `description: null`, `homepage: null`). README title *"LLMs for Graph
  Anomaly Detection"*, authors listed as *"Lumingyuan Tang, Chen Peng, Xingjian
  Dong, Zhiyu Ni"*. **No associated paper, venue or DOI found.** This is a
  student/independent project, **not a publication**.
- It ships `GNN_Methods/rid_mapping.pkl`, `test_index2id.py`,
  `yelp_test_ids.pkl` — i.e. an explicit node-index → review-ID table — and its
  `Data/` folder holds "the orignal text-based dataset from YelpCHI and Amazon
  as well as the preprocessing code" (README, verbatim).
- Sibling evidence `research/_evidence/loop1_c_yelp_text.md:176-258` inspected
  its preprocessing and reports the join is **"PURELY POSITIONAL … raw
  line-number correspondence, and the hotel file is assumed to come first"**,
  with **"Does it prove alignment? NO."** It *"assumes its ordering equals the
  `.mat` ordering and ships `rid_mapping.pkl` as if authoritative"*
  (`loop1_c_yelp_text.md:235-248`).
- Sibling C further measured that naive file order is **wrong**: *"Only 12 of
  45,954 indices actually agree"* (`loop1_c_yelp_text.md:447`).

**Classification: (b) assumed row order, and the assumption is measurably
false.** This is the strongest cautionary datapoint in the whole task.

### C4.4 — LLM-based fraud-detection papers checked, and what each actually did

| Work | Uses YelpChi/Amazon? | What it did about text | Category |
|---|---|---|---|
| **FLAG** (KDD'25, 10.1145/3711896.3737220) | **No** | Declared them text-free and built Reddit/Instagram instead | n/a (excluded) |
| **LGSPF / "Let Relations Speak"** (arXiv:2605.28524) | Yes — Amazon, YelpChi, S-FFSD | **No raw text at all.** Verbatim: *"Following the preprocessing paradigms outlined by Dou et al. (2020a) and Xiang et al. (2023), all unstructured attributes are discarded to enforce the Weak-TAG setting."* Its only "text" is researcher-authored **relation** strings, e.g. *"`<\|graph_pad_relation1\|>`: U-P-U relation embedding — connects users who reviewed the same product."* It reconfirms *"the hand-crafted input feature dimensions for the Amazon, YelpChi, and S-FFSD datasets are rigorously specified as 25, 32, and 126"* | **(d)** — verbalises relations, never touches review text; **no mapping attempted** |
| **DGP** (arXiv:2507.21653, LLM + graph, ByteDance) | **No — it does not use the CARE-GNN `.mat` at all** | Rebuilds from raw. Verbatim: *"Instead of using the handcrafted features introduced in the original work, we directly utilize the original texts for LLM-based methods."* Its Amazon is the **Video** category (not Musical Instruments), node = a **review** labeled helpful/unhelpful — a different graph from `Amazon.mat`. No alignment claim, no alignment verification | **(c)** — rebuilt from raw data from scratch |
| **GADBench** (arXiv:2306.12251) | Yes | Numeric features only; no text | **(d)** |
| **SEFraud** (arXiv:2406.11389) | Yes | Numeric features only; no text | **(d)** |
| `zyni2001/Anomaly-detection-LLM` (unpublished) | Yes | Positional join, shipped as `rid_mapping.pkl` | **(b)** |

**Pattern, and it is uniform:** every *published* work either (c) rebuilds a
different graph from raw text, or (d) never uses text on these graphs at all.
**Zero published works fall in category (a) "proved alignment by
reconstruction."**

### C4.5 — A structural reason no naive mapping can be right for YelpChi

- Rayana & Akoglu's YelpChi release: **67,395 reviews**, "for a set of hotels and
  restaurants in the Chicago area", from **38,063 reviewers** and **201 hotels
  and restaurants**, including "product and user information, timestamp, ratings,
  and a plaintext review" (https://shebuti.com/yelpchi-dataset/, verbatim).
- `YelpChi.mat`: **45,954** nodes.

`45,954 != 67,395`. The `.mat` is a **filtered subset** (~68%) under a selection
rule CARE-GNN never documented in the paper. Any positional join is therefore
**guaranteed** to be misaligned unless the selection rule is reproduced first.
Sibling C reports recovering a product-degree prune (`>800`) that hits 45,954 and
39,277/6,677 simultaneously (`loop1_c_yelp_text.md:414-434`) — that is
**reconstruction evidence, produced by us in this loop, not published by anyone**,
and C itself grades it **"PLAUSIBLE-TO-PROBABLE, NOT PROVEN"**
(`loop1_c_yelp_text.md:561`). **I independently concur with that grading** and
record it as **UNKNOWN-pending-proof**, not as a mapping.

### C4.6 — Bottom line for Q4

- Published, verified mapping: **does not exist.** (AGREE across all sources.)
- Published, *assumed* mapping: exists only in an **unpublished** repo, and its
  assumption is **measurably wrong** (12/45,954 agreement).
- Therefore: **no claim of an "exact" or "canonical" node↔review mapping may be
  made in this project**, and any text attached to these graphs must be stamped
  `text_source: engineered_or_external`, `alignment: UNPROVEN`, in line with the
  repo's existing Phase-9 integrity rule (`research/dataset_notes.md:328-345`).

---

## Origin papers (brief, per the task's "one paragraph each")

**YelpChi.** The dataset originates with **Shebuti Rayana and Leman Akoglu,
"Collective Opinion Spam Detection: Bridging Review Networks and Metadata",
KDD 2015** (the SpEagle paper). Their release is "67,395 reviews for a set of
hotels and restaurants in the Chicago area", 38,063 reviewers, 201 businesses,
with "product and user information, timestamp, ratings, and a plaintext review"
(https://shebuti.com/yelpchi-dataset/). Labels are Yelp's own
filtered-vs-recommended verdict, used as a proxy for spam. The **32 handcrafted
features** are Table 2 of that paper — CARE-GNN's author states the composition
verbatim as "15 review behavior&text features, 9 user behavior features, and 8
product behavior features"
(https://github.com/YingtongDou/CARE-GNN/issues/2#issuecomment-692061862).
**The multi-relational graph form (R-U-R / R-T-R / R-S-R) is NOT Rayana &
Akoglu's** — it was introduced by CARE-GNN, which also chose the 45,954-node
subset. Distribution is email-gated via the author's page and ODDS
(`https://odds.cs.stonybrook.edu/yelpchi-dataset/` — **unreachable from this
host: TLS `SSLV3_ALERT_HANDSHAKE_FAILURE`; UNKNOWN content**).

**Amazon.** Three-link chain, none of which is CARE-GNN. The **corpus** is
**McAuley & Leskovec (2013)**'s Amazon review dump. The **labelling heuristic**
(helpfulness votes as a fraud proxy) comes from **Kumar et al., "REV2:
Fraudulent user prediction in rating platforms", WSDM 2018**. The **25
handcrafted user features** and the ≥20-total-votes filter come from **Zhang,
Wu, Yuan, Yin, Sheng, "GCN-Based User Representation Learning for Unifying
Robust Recommendation and Fraudster Detection" (GraphRfi), SIGIR 2020,
arXiv:2005.10150**, who used a 0.7/0.3 ratio on *Movies & TV*. **CARE-GNN
(Dou et al., CIKM 2020)** then assembled the benchmark we hold: it switched to
*Musical Instruments*, tightened the ratio to **0.8/0.2** — verbatim: *"we label
users with more than 80% helpful votes as benign entities and users with less
than 20% helpful votes as fraudulent entities"*
(https://ar5iv.labs.arxiv.org/html/2008.08692) — and built U-P-U / U-S-U / U-V-U.
CARE-GNN is thus the *constructor* of the graph but the *originator* of neither
the features nor the labels. See `research/_evidence/loop1_b_amazon_raw.md:288-330`
for the sibling agent's independent derivation of the same chain; **we AGREE**.

---

## Unresolved / UNKNOWN register

| Item | Status |
|---|---|
| Precise §/page of FLAG's "lack textual information" sentence | **UNKNOWN** (quote verified twice locally; location not recorded) |
| Any real published source stating Amazon has **24** features | **UNKNOWN — none found.** Premise unconfirmed |
| Exact per-column semantics of `Amazon.mat`'s 25 columns | **UNKNOWN** (source Table 1 lists feature *families*, not 25 atomic columns) |
| Meaning of Amazon fraud-rate denominator in third-party tables | **CONFLICT** (9.5% labeled-only vs 6.9% all-nodes) |
| Which split each published YelpChi/Amazon number used (DGL 70/10/20 seed 717 vs CARE-GNN 40/60 seed 2) | **UNKNOWN per-paper**, frequently unstated |
| CARE-GNN's undocumented 45,954-of-67,395 selection rule | **UNKNOWN as published**; candidate rule reconstructed by sibling C, unproven |
| Verified `.mat` row ↔ review/reviewer ID mapping | **DOES NOT EXIST in any publication** |
| ODDS YelpChi page contents | **UNKNOWN** — TLS handshake failure from this host |
