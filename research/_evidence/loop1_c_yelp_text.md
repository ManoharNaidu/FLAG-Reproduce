# Loop 1 / Subagent C — YelpChi original review TEXT: provenance and alignability

Investigation date: **2026-09-17/18 UTC** (downloads timestamped `2026-09-17T23:11:29Z`).
Scope: research only. No existing repo code modified. `data/raw/*.mat` **not opened** (sibling agent owns it).
All `.mat` invariants below are the ones supplied in the task brief: 45,954 nodes, 39,277 benign / 6,677 spam,
R-U-R 98,630 / R-T-R 1,147,232 / R-S-R 6,805,486 directed entries.

---

## 0. Headline result

The original YelpChi review text **is obtainable**, and the 45,954 graph nodes are **reconstructible from raw
metadata**. Reconstructing the node set and two of the three relation graphs from raw metadata alone reproduces
the `.mat` invariants **exactly**:

| Invariant | Reconstructed from raw metadata | Expected (`.mat`) | Match |
|---|---|---|---|
| Node count | 45,954 | 45,954 | **YES** |
| Benign / spam | 39,277 / 6,677 | 39,277 / 6,677 | **YES** |
| R-U-R directed entries | 98,630 | 98,630 | **YES** |
| R-S-R directed entries | 6,805,486 | 6,805,486 | **YES** |
| R-T-R directed entries | 412,824 | 1,147,232 | **NO — UNRESOLVED** |

The recovered node-selection rule is: **drop the 19 products with more than 800 reviews**.

Verdict: alignment is **plausible-to-probable but NOT YET PROVEN**. See §7.

---

## 1. Authoritative source — URLs tried, verbatim status

### 1.1 Shebuti Rayana homepage — REACHABLE, but download is email-gated

`https://shebuti.com/yelpchi-dataset/`

```
HTTP/1.1 200 OK
Date: Thu, 17 Sep 2026 22:46:11 GMT
Content-Type: text/html; charset=UTF-8
Server: cloudflare
CF-Ray: a3cba53f5fbca440-SYD
```

Content states, verbatim:

> "67,395 reviews for a set of hotels and restaurants in the Chicago area"
> "product and user information, timestamp, ratings, and a plaintext review"
> "201 hotels and restaurants by 38,063 reviewers"
> "In this dataset, there exist 13.23% filtered reviews by 20.33% spammers."

Download instruction, verbatim:

> "To get the datasets with ground truth please email: srayana@cs.stonybrook.edu"

**There is no direct download URL on the author page.** The authoritative release is email-gated.

Primary citation given on that page is **not** SpEagle but:

> Mukherjee et al., "What Yelp fake review filter might be doing?" ICWSM, 2013.

with Rayana & Akoglu KDD 2015 (SpEagle) and SDM 2016 listed as related work.

**Correction to the task brief's background:** the brief said the dataset "originates from Rayana & Akoglu
(KDD 2015)". Per the author's own page, YelpChi **originates from Mukherjee et al. ICWSM 2013**; Rayana &
Akoglu redistributed it and contributed the 32 behavioural features. This is corroborated by the restaurant
text filename shipped in the mirror (`output_review_yelpResData_NRYRcleaned.txt` — Mukherjee's
`yelpResData` / NR=Not-Recommended, YR=Yelp-Recommended naming).

### 1.2 ODDS / Stony Brook — UNREACHABLE from this host (TLS failure)

`https://odds.cs.stonybrook.edu/yelpchi-dataset/`

WebFetch:
```
error:10000410:SSL routines:OPENSSL_internal:SSLV3_ALERT_HANDSHAKE_FAILURE
```

curl:
```
curl: (35) schannel: next InitializeSecurityContext failed: SEC_E_ILLEGAL_MESSAGE (0x80090326)
- This error usually occurs when a fatal SSL/TLS alert is received (e.g. handshake failed).
```

openssl s_client (default, and again with `-tls1_2 -cipher 'DEFAULT:@SECLEVEL=0'`):
```
Connecting to 198.71.233.214
CONNECTED(000001C4)
0C4E0000:error:0A000410:SSL routines:ssl3_read_bytes:ssl/tls alert handshake failure:
  ../openssl-3.2.1/ssl/record/rec_layer_s3.c:865:SSL alert number 40
---
no peer certificate available
```

Plain HTTP reaches Cloudflare but only redirects into the failing HTTPS endpoint:
```
HTTP/1.1 301 Moved Permanently
Location: https://odds.cs.stonybrook.edu/yelpchi-dataset/
Server: cloudflare
CF-RAY: a3cba67e0f75561f-SYD
```

This is **not** the expired-certificate problem noted for `jmcauley.ucsd.edu`; it is a server-side TLS
alert 40 (handshake_failure) with **no certificate presented at all**. ODDS content is therefore **UNKNOWN**
to this investigation — I could not read its file listing and do not assert one. Search-engine snippets
indicate ODDS hosts YelpChi/YelpNYC/YelpZip pages, but I could not verify their contents directly.

### 1.3 SpEagle paper's stated data URL

**UNKNOWN.** I did not retrieve the KDD 2015 paper body. The CARE-GNN CIKM'20 PDF was fetched
(`https://penghao-bdsc.github.io/papers/cikm20.pdf`, 2.3 MB) but could not be parsed — `pypdf` is not
installed in the available interpreter, and WebFetch returned:
`"The provided content is a binary PDF file stream that cannot be parsed as readable text."`
**I therefore did NOT verify the R-U-R / R-T-R / R-S-R definitions from the CARE-GNN paper text.**
The definitions used below were verified from *code* (§2), not from the paper.

### 1.4 `metadata` + `reviewContent` two-file release

The task brief anticipated a `YelpChi.zip` containing files named `metadata` and `reviewContent`.
The mirror I obtained ships `metadata` (as `metadata.gz` / `metadata.txt`) but the review text arrives as
**two** files split by business type (`raw_text.txt` for hotels, `output_review_yelpResData_NRYRcleaned.txt`
for restaurants), not as a single `reviewContent`. Whether the canonical author zip uses a single
`reviewContent` file is **UNKNOWN** (could not read ODDS; author page gives no file listing).

---

## 2. Secondary mirror: `github.com/zyni2001/Anomaly-detection-LLM`

Repo: "LLMs for Graph Anomaly Detection" — Lumingyuan Tang, Chen Peng, Xingjian Dong, Zhiyu Ni.
Treated as **evidence only**; every claim below was independently re-derived from the bytes.

### (a) Files shipped — verbatim GitHub API listing of `Data/`

```
file        171646 Data/PRUNED_DATA_prod-ID_usr-ID_rating_label_review.json
file           943 Data/README.md
file      52658026 Data/UNPRUNED_DATA_prod-ID_usr-ID_rating_label_review.json
file           497 Data/YelpChi_feature_name.txt
file          9785 Data/amazon_preprocess.py
file          8971 Data/helper.py
file        409303 Data/metadata.gz
file       1776502 Data/metadata.txt
file       6191709 Data/musical_reviews.pickle
file      45746857 Data/output_review_yelpResData_NRYRcleaned.txt
file        170833 Data/prod-ID_usr-ID_rating_label_review.json
file       5062002 Data/raw_text.txt
file       2460495 Data/reviews_Musical_Instruments_5.json.gz
file          7946 Data/yelp_preprocess.py
```
plus `GNN_Methods/rid_mapping.pkl` (798,636 bytes) and test/train pickles.

### (b) Schemas

`metadata.gz` / `metadata.txt` — documented in code and **confirmed by parsing**:

> ```python
> # file format: each line is a tuple (user id, product id, rating, label, date)
> items = line.strip().split()
> u_id = items[0]; p_id = items[1]; rating = float(items[2]); label = int(items[3]); date = items[4]
> ```

Measured: 67,395 rows, **every row exactly 5 whitespace-separated fields**, date format `%Y-%m-%d`.
`metadata.txt` is **byte-identical** to `gunzip(metadata.gz)` (both 1,776,502 bytes) — verified.

Emitted JSON schema, from `Data/README.md`, verbatim:

> ```
> {prod_ID: user_ID, user_label, review_label, rating score, review text}
> ```
> "`review_label` - Created by Yelp company. For label value 1 means non-spam review and -1 means spam
> review. There are 13.23% spam reviews in total in the unpruned data."
> "`user_label` - Created by us. For label value 1 means normal users (authors with no spam reviews) and -1
> means fraud users (authors with at least one spam review). There are 20.33% spam users in total."

Note the JSON is keyed **by product**, so it is a product→reviews grouping, not a node-indexed table.

### (c) **The ordering assumption when joining text to graph nodes — PURELY POSITIONAL**

This is the critical finding. From `Data/helper.py`, verbatim:

```python
def load_text_data(hot_text_name, res_text_name, removed_user, removed_prod, review_ground_truth):

	# load raw text
	text_list = []
	with open(hot_text_name, 'rt') as f_in:
		for line in f_in.readlines():
			text_list.append(line)
	with open(res_text_name, 'rt') as f_in:
		for line in f_in.readlines():
			text_list.append(line)

	# match review text and review id
	review_text_mapping = {}
	line_index = 0
	with open(graph_meta_name, 'rt') as f_in:
		for line in f_in.readlines():
			line = line.split()
			u_id = line[0]
			p_id = line[1]

			if u_id not in removed_user and p_id not in removed_prod:
				review_text_mapping[(u_id, p_id)] = text_list[line_index]

			line_index += 1
```

with

```python
	hot_text_name = 'raw_text.txt'
	res_text_name = 'output_review_yelpResData_NRYRcleaned.txt'
```

**The assumption is: line `i` of `metadata.txt` corresponds to element `i` of
`concat(raw_text.txt, output_review_yelpResData_NRYRcleaned.txt)`.** There is no id, no checksum, no
content check — it is raw line-number correspondence, and the hotel file is assumed to come first.

Note also `line_index += 1` is **outside** the filter branch (correct — the counter tracks the full file),
so the positional logic is at least self-consistent.

The node-index assignment, from `Data/yelp_preprocess.py`, verbatim:

```python
    # map review id to adj matrix id
    rid_mapping = {}
    r_index = 0
    for review in review_ground_truth.keys():
        rid_mapping[review] = r_index
        r_index += 1
```

`review_ground_truth` is built by iterating `user_data.items()` — i.e. **grouped by user in first-appearance
order**, *not* metadata file order. I confirmed this empirically (§4).

### (d) **Does it prove alignment? NO.**

`yelp_preprocess.py` builds its **own** adjacency from the metadata and saves its **own** `rid_mapping.pkl`:

```python
    review_adj, rid_mapping = meta_to_homo(meta_data_name)
    with open('../GNN_Methods/rid_mapping.pkl', 'wb') as pkl_file:
        pickle.dump(rid_mapping, pkl_file)
    # sp.save_npz(review_adj_name, review_adj)
```

**At no point does the repository load `YelpChi.mat` and compare anything against it.** There is no
adjacency equality check, no label-vector check, no feature check, no row-count assertion. The repo
*assumes* its ordering equals the `.mat` ordering and ships `rid_mapping.pkl` as if authoritative.
Its `GNN_Methods/test_index2id.py` simply inverts that mapping to name test rows:

```python
index_to_id = {v: k for k, v in rid_mapping.items()}
test_ids = [index_to_id[idx] for idx in idx_test if idx in index_to_id]
```

So the repo is **evidence of a plausible recipe, not proof of alignment**. The proof had to be
constructed independently — which is what §4 does.

### (e) Relation definitions — verified **from this code**, not from the paper

`yelp_preprocess.py` implements all five candidate relations (four commented out), verbatim:

```python
    # 1) r-product-r
    # 2) r-user-r
    # for u, reviews in user_prod_graph.items():
    # 	for r0 in reviews:
    # 		for r1 in reviews:
    # 			if r0[0] != r1[0]:
    # 				review_adj[rid_mapping[(u, r0[0])], rid_mapping[(u, r1[0])]] = 1
    # 3) r-time-r
    for p, reviews in prod_user_graph.items():
        for r0 in reviews:
            for r1 in reviews:
                if r0[0] != r1[0] and time_judge(r0[3], r1[3]) == True:
                    review_adj[rid_mapping[(r0[0], p)], rid_mapping[(r1[0], p)]] = 1
    # # 4) r-star-r
    # for p, reviews in prod_user_graph.items():
    # 	for r0 in reviews:
    # 		for r1 in reviews:
    # 			if r0[0] != r1[0] and r0[1] == r1[1]:
    # 				review_adj[rid_mapping[(r0[0], p)], rid_mapping[(r1[0], p)]] = 1
```

and

```python
def time_judge(time1, time2):
    date1 = datetime.strptime(time1, '%Y-%m-%d')
    date2 = datetime.strptime(time2, '%Y-%m-%d')
    if date1.year == date2.year and date1.month == date2.month:
        return True
```

So this code's definitions are:
- **R-U-R** = same user (no product constraint)
- **R-S-R** = same product AND same star rating
- **R-T-R** = same product AND same (year, month)

These match the definitions stated in the task brief. Two of them reproduce the `.mat` exactly; the third
does not (§5).

### (f) Two different prune thresholds ship in the same repo

`yelp_preprocess.py` (graph construction):
```python
	for prod, reviews in prod_user_graph.items():
		if len(reviews) > 800:
			removed_prod.append(prod)
```
`helper.py` (LLM-prompting subset):
```python
def remove_reviews(upg, pug, threshold=20, prune=True):
```
The **>800** rule is the one that reproduces the `.mat` node count. The `threshold=20` rule produces the
small `PRUNED_DATA_*.json` used for LLM prompting and is **not** the graph node set. Anyone reusing this
repo must not confuse the two.

### (g) Other repositories checked

`github.com/RohanMukka/Combining-Transformer-Semantics-and-Reviewer-Behavior-for-Fake-Review-Detection-on-Yelp`
README claims, verbatim:

> "YelpCHI (Rayana & Akoglu, KDD 2015): 45,954 reviews, 6,677 fake (14.53%), 39,623 reviewers, 1,224
> businesses. This project uses the CSV distribution carrying review text, reviewer and business
> identifiers, star ratings, and labels."

**This claim is partly WRONG and should not be relied on.** The review/fake counts are right, but YelpChi
has **201** businesses in total (verified from metadata) and the 45,954-node subset has **182** products
and **29,431** users — not "1,224 businesses / 39,623 reviewers". No download URL is given. Flagging as
an unreliable secondary source.

No repository was found that loads `YelpChi.mat` and validates a text join against it.
Searching for repos pairing `reviewContent` + `metadata` + `YelpChi.mat` returned **no** relevant hit.

---

## 3. The join-key situation

**There is no explicit review-id column on either side.** Route (a) from the brief is therefore **dead**:

- Raw metadata columns are exactly `user_id prod_id rating label date` — 5 fields, no id. Verified: the
  field-count histogram over all 67,395 rows is `{5: 67395}`.
- The `.mat` is understood to carry only features/labels/adjacency (per the sibling agent's remit).

However, **(user_id, prod_id) is a valid surrogate review key**: measured 67,395 unique pairs across
67,395 rows — **zero collisions**. So a review is uniquely identified by its (user, product) pair, and the
whole problem reduces to recovering the **permutation** from pairs to `.mat` row indices.

That makes route (b) — **reconstructing the relation graphs and matching them against the `.mat`** — the
viable route. Route (c) (matching the 32 features) is a weaker fallback: `YelpChi_feature_name.txt` states
verbatim

> "% Please ask the authors of the above paper for the feature generating code"

so the features are **not** recomputable from raw metadata without unavailable code, and several (RD, ETF,
ISR, DL_u, DL_b, …) depend on text models. Route (c) is impractical. Some features are trivially
recoverable though — e.g. `8 L` (review length) and `6 PCW`/`7 PC` (word/char counts) are computable from
the text and could serve as an **independent cross-check** (see §7).

---

## 4. Independent reconstruction — what I actually verified

All numbers below were computed by me from the downloaded raw files, not taken from any repo or paper.

### 4.1 Raw release statistics

```
metadata.txt bytes: 1776502   metadata.gz decompressed bytes: 1776502
IDENTICAL CONTENT: True
metadata rows (non-empty): 67395
first 3 rows: ['201 0 5.0 1 2011-06-08', '202 0 3.0 1 2011-08-30', '203 0 5.0 1 2009-06-26']
last row: 38263 200 5.0 1 2010-01-25
field-count histogram: {5: 67395}
unique users: 38063 unique products: 201
label counts: {'1': 58476, '-1': 8919}
rating counts: {'1.0': 3493, '2.0': 5003, '3.0': 9186, '4.0': 24314, '5.0': 25399}
date min/max: 2004-10-12 2012-10-08
unique (user,prod) pairs: 67395  pairs appearing >1x: 0  extra rows lost by pair-keying: 0
```

38,063 users / 201 products / 8,919 spam (= 13.23%) **exactly** matches shebuti.com's stated figures.
This authenticates the mirror as the genuine YelpChi release.

**Label encoding: `1` = genuine (Yelp-recommended), `-1` = spam (Yelp-filtered).** Note this is *not* the
0/1 encoding of the `.mat`; `create_ground_truth` remaps `-1 → 1 (spam)`.

### 4.2 Text files line up dimensionally and semantically

```
metadata rows=67395  hotel lines=5854  restaurant lines=61541  hotel+rest=67395
DIMENSIONS CONSISTENT: True
distinct products in first 5854 rows: 72   in remaining rows: 129   overlap: 0
first-block product ids (sorted int): [0..11] ...max 71
rest-block product id min: 72
```

This is strong corroboration of the positional join, beyond mere row-count equality:
the hotel/restaurant **file boundary at line 5,854 lands exactly on the product-id boundary 71→72**, with
**zero** product overlap between the two blocks. A wrong offset would not produce a clean partition.

Spot checks confirm content type:
- hotel line 0: *"…there are two kinds of people, those who will give the **Tokyo Hotel** 5 stars…"* → a hotel.
- restaurant line 0, at metadata row 5854 = `['5227','72','5.0','1','2012-09-22']` (first restaurant,
  product 72): *"Unlike **Next**, which we'd eaten at the previous night… **Alinea** delivers a meal…"* →
  a Chicago restaurant.

This is evidence *for* the positional join, but it is **block-level**, not row-level. It does not exclude a
permutation *within* a block.

### 4.3 Recovering the node-selection rule

Sweeping product-degree thresholds against the target 45,954:

```
product review-count: min 1 max 2159
prod-degree sweep exact hits: prod<=794 ... prod<=807   (all give 45954)
```

The plateau 794–807 contains **800**, matching `yelp_preprocess.py`'s `if len(reviews) > 800`. Applying it:

```
products removed (>800 reviews): 19
['103','104','108','115','119','122','124','129','137','141','162','72','73','75','78','90','95','97','99']
kept reviews: 45954
label split after prune: benign(1)=39277 spam(-1)=6677
EXPECTED from .mat: benign=39277 spam=6677
MATCH: True
users surviving: 29431 products surviving: 182
```

**Both the node count and the exact label split are reproduced.** The joint probability of an unrelated
rule hitting 45,954 *and* 39,277/6,677 simultaneously is negligible.

### 4.4 Row order

```
rid_mapping entries: 45954   rid index range: 0 45953
rid_mapping keyset == pruned(>800) keyset: True
rid_mapping indices equal to metadata-file order:      12 / 45954
rid_mapping indices equal to group-by-user order:   45954 / 45954
```

**The canonical order is group-by-user first-appearance order, NOT metadata file order.** This is a trap:
the first few rows coincide (users 201, 202, 203 all first review product 0), so a naive spot check of the
first handful of rows would wrongly suggest file order. Only **12 of 45,954** indices actually agree.

---

## 5. Relation reconstruction — 2 of 3 exact

Directed entries, no self-loops, under the pruned node set:

```
  R-U-R = 98630    expected .mat 98630    MATCH=True
  R-S-R = 6805486  expected .mat 6805486  MATCH=True
  R-T-R = 412824   expected .mat 1147232  MATCH=False
```

The caller's directed counts are exactly 2× the CARE-GNN published undirected edge counts
(49,315 / 573,616 / 3,402,743), which is consistent and confirms I am comparing like with like.

**R-U-R and R-S-R match to the entry.** That is a very strong structural result: R-S-R in particular is a
6.8-million-entry quantity determined by the joint (product, rating) multiset, and it lands exactly.

### R-T-R remains UNRESOLVED

Target 1,147,232. Roughly 35 definitions were tested; **none** reproduce it. Verbatim results:

```
prod + year-month                    412824
prod + year-quarter                 1212928
prod + year-halfyear                2310578
prod + year                         4448854
prod + month-of-year                1909222
prod + exact date                     15362
GLOBAL year-month (no prod)        34016050
GLOBAL exact date (no prod)         1157812
user + year-month                     14526
prod + rating + year-month           129974
```

Sliding day-windows on the same product (target falls strictly between 42 and 43 days — no integer hit):

```
|dt|<=31 : 843980     |dt|<=41 : 1103882
|dt|<=42 : 1130258     |dt|<=43 : 1156252
|dt|<=44 : 1182020     |dt|<=45 : 1207410
```

Month-index windows on the same product (target falls between 0 and 1 — no integer hit):

```
same prod & |month index diff|<=0 : 412824
same prod & |month index diff|<=1 : 1210758
```

Near-miss variants of the global same-date idea (1,157,812 is tantalisingly close to 1,147,232, a 0.92%
gap, but no adjustment lands exactly):

```
global same-date:                                 1157812
global same-date minus same-user-same-date:       1151186
global same-date minus same-prod-same-date:       1142450
```

**Status: UNKNOWN.** I will not guess. Two readings are possible and I cannot presently distinguish them:

1. The R-T-R construction used for the `.mat` differs from the documented "same product, same month" rule
   in some way I have not recovered.
2. **The `date` column in this mirror differs from the one used to build the `.mat`.** This reading is
   worth taking seriously: the two relations that match (R-U-R, R-S-R) are exactly the two that **do not
   use the date field**, and the only relation that fails is the only one that **does**. That is a
   suspicious pattern, not a coincidence to wave away.

Distinguishing these matters, because reading (2) would mean the mirror's metadata is a *near* but not
*identical* snapshot — which would weaken, though not destroy, the text-alignment claim.

---

## 6. What was downloaded

Into `datasets/raw/yelpchi/`, manifest at `datasets/manifests/yelpchi_source.json`
(source URLs, UTC timestamp, SHA-256, sizes, descriptions):

| file | sha256 (abbrev) | bytes |
|---|---|---|
| `metadata.gz` | `253db76d…ec84` | 409,303 |
| `metadata.txt` | `c674adf2…52c1a` | 1,776,502 |
| `raw_text.txt` | `268502eb…4c798` | 5,062,002 |
| `output_review_yelpResData_NRYRcleaned.txt` | `6fb92e48…bd2808` | 45,746,857 |
| `YelpChi_feature_name.txt` | `e1f9d5bc…f8a4aae` | 497 |
| `rid_mapping.pkl` | `41e71de7…21a179` | 798,636 |

Derived (clearly marked as such in the manifest):
- `yelpchi_aligned_reviews.jsonl` — 45,954 rows, canonical order, fields
  `node_idx, user_id, prod_id, rating, label, date, text` (39,769,264 bytes).
- `alignment_fingerprint.csv` — per-node R-U-R degree, R-S-R degree, label (669,984 bytes).

Fingerprints of the canonical-order vectors, for the sibling agent to check against the `.mat` **without
me touching it**:

```
SHA256 R-U-R degree vector : 97fc404aaf59385537a506bc4315235ed9fc9dfeff8c049655b93390b4ec02ff
SHA256 R-S-R degree vector : d21ce24342e0f837dcedbd7b1d65467ca93e34504524f6f2fa9948943a013dba
SHA256 label vector        : e422ecd60b5a0d2a3945b57aed5abc5e28eb3168875f44c1248fe9c0924769fe
```

The 32 feature names (indices 0–31) are in `YelpChi_feature_name.txt`: 0–14 review features
(`Rank, RD, EXT, DEV, ETF, ISR, PCW, PC, L, PP1, RES, SW, OW, DL_u, DL_b`), 15–23 user features
(`MNR, PR, NR, avgRD, WRD, BST, ERD, ETG, RL`), 24–31 product features
(`MNR, PR, NR, avgRD, WRD, ERD, ETG, RL`). Note the file's own header labels these as "15-24 user" and
"24-31 product" while listing 15–23 and 24–31 — an **off-by-one in the source file's comment**; the
numbered entries are authoritative and total 32.

---

## 7. Verdict on EXACT alignment, and what would close it

**Current status: PLAUSIBLE-TO-PROBABLE, NOT PROVEN.** Not yet safe to call EXACT.

Evidence **for**:
1. Node count 45,954 reproduced exactly by a documented prune rule (>800).
2. Label split 39,277 / 6,677 reproduced exactly.
3. R-U-R = 98,630 exact.
4. R-S-R = 6,805,486 exact.
5. Text files partition exactly on the hotel/restaurant product boundary with zero overlap; content type
   matches on inspection.
6. `(user_id, prod_id)` is a collision-free key over all 67,395 rows.
7. A canonical row order exists and is reproducible (group-by-user), matching the shipped `rid_mapping.pkl`
   on 45,954/45,954 rows.

Evidence **against / still open**:
1. **R-T-R does not reproduce** under ~35 tested definitions — and it is the *only* date-dependent
   relation, raising the possibility the mirror's `date` column is not the one used for the `.mat`.
2. All matched quantities so far are **aggregate totals**, not per-row. Equal totals do not prove equal
   ordering. The ordering claim currently rests on a third-party pickle whose author never verified it.
3. The **within-block** text↔metadata correspondence is assumed, never checked at row level. §4.2 proves
   the block boundary, not the interior.
4. The authoritative author release was never obtained (email-gated; ODDS unreachable), so the mirror
   itself is unverified against the source-of-truth bytes.

**Three concrete checks would upgrade this to EXACT** (all cheap, none requiring new downloads):

- **C1 — per-row degree match.** Compare `alignment_fingerprint.csv` row-by-row against per-row degrees
  of `net_rur` and `net_rsr` in `YelpChi.mat`. If all 45,954 rows agree for both relations, the
  permutation is pinned. This is the single highest-value test and it is decisive.
- **C2 — label vector match.** Compare the canonical-order label vector element-by-element against the
  `.mat` label array (remapping `1→0` benign, `-1→1` spam). Cheap, and independent of C1.
- **C3 — text↔row sanity via a recoverable feature.** Feature index `8 L` (review length) and `6 PCW` /
  `7 PC` are computable directly from the joined text. Correlating them against the corresponding `.mat`
  feature columns in canonical order would independently validate the *within-block* positional text join
  — the one link §4.2 does not cover. If `L` matches per-row, the text join is proven end to end.

Until C1/C2 pass, treat the mapping as **high-confidence provisional**. Do not publish reproduction
numbers that depend on node↔text identity without stating this caveat. If C1 passes but the R-T-R question
remains open, the alignment is still sound for text purposes (R-T-R affects graph structure, not the
review-to-node identity), but the discrepancy should be documented.

---

## 8. Version discrepancies (task 5)

Precise attribution of counts to releases:

| Release | Reviews | Entities | Status |
|---|---|---|---|
| **YelpChi — raw** | **67,395** | 201 businesses (72 hotels + 129 restaurants), 38,063 reviewers | Verified by me from downloaded metadata; matches shebuti.com verbatim |
| **YelpChi — graph / `.mat`** | **45,954** | 182 products, 29,431 users | Verified by me: raw minus 19 products with >800 reviews. 6,677 spam (14.53%), 39,277 benign |
| **YelpNYC** | 359,052 | restaurants in New York City | Per shebuti.com/ODDS search results — **different dataset**, not a YelpChi variant |
| **YelpZip** | 608,598 | 5,044 restaurants, 260,277 reviewers (NY/NJ/VT/CT/PA by zipcode) | Per shebuti.com/ODDS search results — **different dataset** |

Key point for this project: **67,395 vs 45,954 is not a version discrepancy, it is a documented
subsetting.** The raw release is the superset; the fraud-graph literature uses the >800-pruned subset.
Papers quoting "YelpChi = 67,395 reviews" and "YelpChi = 45,954 nodes" are both correct about different
artefacts. The 13.23% spam rate belongs to the **raw** release; **14.53%** belongs to the **pruned graph**.

**45,941:** the brief asked about this count. **UNKNOWN** — I found no source attributing 45,941 to any
YelpChi release, and I did not reproduce it from any threshold in the sweep. Not asserting it exists.

---

## 9. Items explicitly labelled UNKNOWN

- Contents and file listing of the ODDS / Stony Brook YelpChi page — **unreachable** (TLS alert 40).
- The SpEagle (KDD 2015) paper's stated data URL and its verbatim relation definitions — **not retrieved**.
- CARE-GNN CIKM'20 verbatim relation definitions — PDF fetched but **unparseable** in this environment.
- Whether the canonical author zip contains a single `reviewContent` file — **unverified**.
- The correct R-T-R construction rule — **unresolved** after ~35 candidate definitions.
- Whether the mirror's `date` column is byte-identical to the one used to build `YelpChi.mat` — **unknown**,
  and currently the leading suspect for the R-T-R gap.
- Redistribution/licence terms for the review text — **not located**.
- Provenance of the count 45,941 — **no source found**.
