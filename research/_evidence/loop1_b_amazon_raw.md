# Loop 1 / Subagent B — Amazon Musical Instruments RAW REVIEW provenance

**Scope:** raw review corpus only. `data/raw/*.mat` and `experiments/yelpchi_amazon/` untouched (sibling agent).
**Date:** 2026-09-17/18 (UTC timestamps inline). **Python:** `/c/Users/hp/miniforge3/envs/dgp-bl-consisgad/python.exe` (3.9).
**Repo code modified:** none. **Files added:** `datasets/raw/amazon_musical/*`, `datasets/manifests/amazon_source.json`, this file.

---

## HEADLINE RESULT

The Amazon fraud benchmark was **NOT** built from any 5-core file. It was built from the **full, non-5-core 2014 file
`reviews_Musical_Instruments.json.gz`** (500,176 reviews / 339,231 reviewers), filtered to users with **>= 20 total
helpful-votes**, then labelled by helpful-vote ratio.

Applying that rule reproduces the benchmark's label counts **exactly**:

| quantity | reconstructed | Amazon.mat target | match |
|---|---|---|---|
| fraud (ratio < 0.20) | **821** | 821 | EXACT |
| benign (ratio > 0.80) | **7,818** | 11,123 - 3,305 = 7,818 | EXACT |
| labeled total | **8,639** | 8,639 | EXACT |
| cohort size (nodes) | 11,447 | 11,944 | -497 |
| unlabeled / ambiguous | 2,808 | 3,305 | -497 |

Adjacent thresholds are not close, so this is not a coincidence:

```
sum_tv>=19   N= 12273( +329)  benign= 8283( +465)  fraud= 894( +73)
sum_tv>=20   N= 11447( -497)  benign= 7818(   +0)  fraud= 821(  +0)   <-- EXACT on both labels
sum_tv>=21   N= 10700(-1244)  benign= 7373( -445)  fraud= 765( -56)
```

The `>= 20 votes` threshold was then **confirmed verbatim in the primary source** (Zhang et al. 2020, section 3.1) —
it was discovered empirically first and corroborated textually second. See section 5.

**Residual: UNKNOWN.** 497 nodes are unaccounted for and all 497 sit in the ambiguous/unlabeled band. Union
hypotheses (`sum_tv>=20 | nrev>=k` for k in {20,25,30,35,40,50,60,80,100}) were tested and none closes the gap while
preserving the exact label counts. Most likely a slightly different snapshot of the raw dump (the SNAP file is dated
2016-02-18) or an undocumented extra inclusion criterion. Not fabricated, not resolved.

---

## 1. Locating the canonical file

### 1.1 The canonical host is genuinely broken

```
$ curl -sS -I --max-time 30 "https://jmcauley.ucsd.edu/data/amazon/"
curl: (35) schannel: next InitializeSecurityContext failed: SEC_E_CERT_EXPIRED (0x80090328) - The received certificate has expired.
```

Independently confirmed with OpenSSL:

```
$ echo | openssl s_client -connect jmcauley.ucsd.edu:443 -servername jmcauley.ucsd.edu
depth=0 C=US, ST=California, O=University of California, San Diego, CN=jmcauley.ucsd.edu
verify error:num=10:certificate has expired
notAfter=May 21 23:59:59 2026 GMT
---
 0 s:C=US, ST=California, O=University of California, San Diego, CN=jmcauley.ucsd.edu
   i:C=US, O=Internet2, CN=InCommon RSA Server CA 2
   v:NotBefore: May 21 00:00:00 2025 GMT; NotAfter: May 21 23:59:59 2026 GMT
```

Certificate expired **2026-05-21**, i.e. ~4 months before this investigation. The host is alive (IP 137.110.160.73)
but its TLS is unusable without `--insecure`. **`--insecure` was never used.** It was not needed.

### 1.2 Host status table (verbatim)

| host | status |
|---|---|
| `https://jmcauley.ucsd.edu/data/amazon/` | `curl: (35) schannel: ... SEC_E_CERT_EXPIRED (0x80090328) - The received certificate has expired.` — **NOT USED** |
| `https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html` | `HTTP/1.1 200 OK` (`Server: Apache/2.4.29 (Ubuntu)`, 37,893 B) |
| `https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html` | `HTTP/1.1 200 OK` (57,097 B) |
| `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/...` | `HTTP/1.1 200 OK` |
| `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/...` | `HTTP/1.1 200 OK` (`Strict-Transport-Security: max-age=15768000`) |
| `https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon/` | `HTTP/1.1 404 Not Found` |
| `https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023` | `HTTP/1.1 200 OK` — wrong vintage, not used |
| Kaggle | **not consulted** |

### 1.3 The provenance chain that replaces the dead host

`cseweb.ucsd.edu/~jmcauley/` is Julian McAuley's own UCSD CSE homespace, served over valid TLS, and it hosts the
*same* dataset landing pages. Those pages' own download links point at SNAP and mcauleylab — so the download hosts are
authoritative by McAuley's own designation, not by third-party assertion.

Header of `links.html`, confirming authorship and that this is the older release kept for reproducibility:

```html
<title>Amazon review data</title>
<meta name="author" content="Julian McAuley">
  <h1>Amazon product data</h1>
  <p><b><a href="http://cseweb.ucsd.edu/~jmcauley/">Julian McAuley</a></b>, UCSD</p>
<h2>Please see the <a href="https://amazon-reviews-2023.github.io/">2023 version of this dataset</a></h2>
<p>This (older) version is mainly here for the sake of reproducing past results</p>
```

### 1.4 Release disambiguation — the filename in the task brief is the 2018 one

From `links.html` (**2014** release), lines 211-214 and 502-503:

```html
<td>Musical Instruments</td>
<td><a href="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz">5-core</a> (10,261 reviews)</td>
<td><a href="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Musical_Instruments.csv">ratings only</a> (500,176 ratings)</td>
...
<td><a href="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments.json.gz">reviews</a> (500,176 reviews)</td>
```

From `amazon_v2/index.html` (**2018** release):

```html
<td><a href=".../categoryFiles/Musical_Instruments.json.gz">reviews</a> (1,512,530 reviews)</td>
<td><a href=".../metaFiles2/meta_Musical_Instruments.json.gz">metadata</a> (120,400 products)</td>
<td><a href=".../categoryFilesSmall/Musical_Instruments_5.json.gz">5-core</a> (231,392 reviews)</td>
<td><a href=".../categoryFilesSmall/Musical_Instruments.csv">ratings only</a> (1,512,530 ratings)</td>
```

**Naming rule:** the `reviews_` prefix marks the **2014** release; the bare name marks the **2018** release.
The brief's filename `Musical_Instruments_5.json.gz` is therefore **the 2018 5-core**, which is the wrong vintage.

---

## 2. Files obtained (all over validated TLS, no cert bypass)

Recorded in `datasets/manifests/amazon_source.json`.

| file | release | bytes | SHA-256 | downloaded (UTC) |
|---|---|---|---|---|
| `reviews_Musical_Instruments.json.gz` | 2014 full | 121,616,883 | `8b46e724abe8da4316b30f51b692843d5dc0abb645d72ad2c44f5aa740d230b0` | 2026-09-17T22:47:48Z |
| `reviews_Musical_Instruments_5.json.gz` | 2014 5-core | 2,460,495 | `03f9cc27ae13a454cb4ea6cb620fb10546cc1194ef921441d1c5f34addff2748` | 2026-09-17T22:47:16Z |
| `Musical_Instruments_5.json.gz` | 2018 5-core | 39,353,085 | `e3ae8a3f0d70bf63242a49539a8847d30216a3aa2938c049c7f43d0473fef1bc` | 2026-09-17T22:47:16Z |

Server `Last-Modified`: 2016-02-18 / 2016-04-26 / 2025-01-17 respectively.
`curl` write-out confirmations: all `http=200`, byte counts equal to the advertised `Content-Length`.

**Provenance caveat: NONE required.** No `--insecure`, no `-k`, no `verify=False`, no Kaggle, no unofficial mirror.

---

## 3. Characterisation

### 3.1 `reviews_Musical_Instruments_5.json.gz` — 2014 5-core

```
n_reviews: 10261
FIELDS (name | count | present% | types | example):
  reviewerID         |    10261 | 100.00% | str    | 'A2IBPI20UZIR0U'
  asin               |    10261 | 100.00% | str    | '1384719342'
  helpful            |    10261 | 100.00% | list   | [0, 0]
  reviewText         |    10261 | 100.00% | str    | "Not much to write about here, but it does exactly what it's supposed to..."
  overall            |    10261 | 100.00% | float  | 5.0
  summary            |    10261 | 100.00% | str    | 'good'
  unixReviewTime     |    10261 | 100.00% | int    | 1393545600
  reviewTime         |    10261 | 100.00% | str    | '02 28, 2014'
  reviewerName       |    10234 |  99.74% | str    | 'cassandra tu "Yeah, well, that's just like, u...'

unique reviewerID: 1429
unique asin: 900
reviews-per-reviewer: min=5 median=6.0 max=42 mean=7.181
reviewers with >=1: 1429 | >=5: 1429 | >=10: 188
unixReviewTime range: 1095465600 .. 1405987200  (2004-09-18 .. 2014-07-22 UTC)
```

Vote metadata: **`helpful: [helpful_votes, total_votes]`** — a 2-element list, i.e. numerator *and* denominator.
Matches McAuley's advertised "10,261 reviews" and the brief's cited "~10,261 reviews / 1,429 reviewers".

### 3.2 `Musical_Instruments_5.json.gz` — 2018 5-core

```
n_reviews: 231392
  overall            |   231392 | 100.00% | float | 5.0
  verified           |   231392 | 100.00% | bool  | True
  reviewTime         |   231392 | 100.00% | str   | '10 30, 2016'
  reviewerID         |   231392 | 100.00% | str   | 'A3FO5AKVTFRCRJ'
  asin               |   231392 | 100.00% | str   | '0739079891'
  unixReviewTime     |   231392 | 100.00% | int   | 1477785600
  reviewerName       |   231367 |  99.99% | str   | 'francisco'
  reviewText         |   231344 |  99.98% | str   | "It's good for beginners"
  summary            |   231341 |  99.98% | str   | 'Five Stars'
  style              |   121310 |  52.43% | dict  | {'Format:': ' Misc. Supplies'}
  vote               |    34777 |  15.03% | str   | '2'
  image              |     3889 |   1.68% | list  | [...]

unique reviewerID: 27530
unique asin: 10620
reviews-per-reviewer: min=3 median=6.0 max=264 mean=8.405
reviewers with >=5: 27520 | >=10: 6203
unixReviewTime range: 1067299200 .. 1538092800  (2003-10-28 .. 2018-09-28 UTC)
```

**Critical field difference:** `helpful` is gone. It is replaced by **`vote`** — a *string* holding only the helpful
count, present on just 15.03% of reviews, with **no total-votes denominator**. The ">80% helpful / <20% helpful"
ratio is therefore **mathematically uncomputable from the 2018 release**. This alone rules out 2018 as the
benchmark's source, independently of any count argument.

Note also `min=3` reviews/reviewer despite the "5-core" label — the 2018 5-core is core-filtered on items, not users.

### 3.3 `reviews_Musical_Instruments.json.gz` — 2014 FULL (the real source)

```
n_reviews: 500176
unique reviewerID: 339231
unique asin: 83046
reviews/reviewer: min=1 median=1.0 max=483 mean=1.4744
  reviewers with >=1 reviews: 339231
  reviewers with >=2 reviews:  68317
  reviewers with >=3 reviews:  29040
  reviewers with >=5 reviews:  10057
  reviewers with >=10 reviews:  2273
date range: 1998-04-25 .. 2014-07-23
```

Same schema as the 2014 5-core, including `helpful: [h, t]`. First record verbatim:

```json
{"reviewerID": "A1YS9MDZP93857", "asin": "0006428320", "reviewerName": "John Taylor", "helpful": [0, 0], "reviewText": "The portfolio is fine except for the fact that the last movement of sonata #6 is missing. What should one expect?", "overall": 3.0, "summary": "Parts missing", "unixReviewTime": 1394496000, "reviewTime": "03 11, 2014"}
```

---

## 4. The critical comparison vs 11,944 nodes

| candidate file | unique `reviewerID` | vs 11,944 | verdict |
|---|---|---|---|
| 2014 5-core `reviews_Musical_Instruments_5.json.gz` | **1,429** | -10,515 | **not at all** — ruled out |
| 2018 5-core `Musical_Instruments_5.json.gz` | **27,530** | +15,586 | **not at all** — ruled out (and no vote denominator) |
| 2014 full `reviews_Musical_Instruments.json.gz` | **339,231** | +327,287 | superset — **correct source, requires filtering** |
| -> filtered to `sum(total_votes) >= 20` | **11,447** | **-497** | **near-exact; labels exact** |

**No raw file matches 11,944 directly.** The graph's node set is a *filtered subset* of the full 2014 corpus, and the
filter is the >=20-total-votes rule. This is the major finding the brief anticipated: the benchmark was **not** built
from a 5-core file.

### 4.1 The threshold sweep (verbatim)

```
target                             N= 11944        benign= 7818       fraud= 821      unlab= 3305
sum_tv>=15                         N= 16787(+4843) benign=10971(+3153) fraud=1218(+397) unlab= 4598(+1293)
sum_tv>=16                         N= 15564(+3620) benign=10304(+2486) fraud=1142(+321) unlab= 4118( +813)
sum_tv>=17                         N= 14280(+2336) benign= 9509(+1691) fraud=1053(+232) unlab= 3718( +413)
sum_tv>=18                         N= 13185(+1241) benign= 8814( +996) fraud= 980(+159) unlab= 3391(  +86)
sum_tv>=19                         N= 12273( +329) benign= 8283( +465) fraud= 894( +73) unlab= 3096( -209)
sum_tv>=20                         N= 11447( -497) benign= 7818(   +0) fraud= 821(  +0) unlab= 2808( -497)
sum_tv>=21                         N= 10700(-1244) benign= 7373( -445) fraud= 765( -56) unlab= 2562( -743)
sum_tv>=25                         N=  8307(-3637) benign= 5852(-1966) fraud= 563(-258) unlab= 1892(-1413)
```

Boundary-condition check (rules out inclusive comparisons): at `sum_tv>=20`, using `>=0.80` gives benign 7,934
(+116) and `<=0.20` gives fraud 842 (+21). So the comparisons are **strictly** `> 0.80` and `< 0.20`, as written in
CARE-GNN.

Alternative aggregations rejected: per-review `max(total_votes) >= k` misses badly for all k
(k=15 -> fraud 1,167; k=20 -> fraud 777). Conjunctions with review count collapse the fraud class
(`sum_tv>=20 & nrev>=2` -> fraud 176).

### 4.2 The 497-node residual — UNKNOWN

Union hypotheses, all tested and all rejected:

```
tv>=20 | nrev>=20    N= 11572( -372) benign= 7868(  +50) fraud= 822(  +1)
tv>=20 | nrev>=30    N= 11469( -475) benign= 7823(   +5) fraud= 822(  +1)
tv>=20 | nrev>=50    N= 11448( -496) benign= 7819(   +1) fraud= 821(  +0)
tv>=20 | nrev>=100   N= 11447( -497) benign= 7818(   +0) fraud= 821(  +0)
```

Any union large enough to add nodes also perturbs the exact label counts. The 497 missing nodes are therefore
**entirely within the ambiguous band** and their origin is **UNKNOWN**. Candidate explanations, none verified:
a marginally different snapshot of the raw dump; users present in the rating CSV but absent from the review JSON;
or an unpublished inclusion step in CARE-GNN's preprocessing.

### 4.3 What the mismatch implies for node <-> reviewer mapping

- A naive positional join against any 5-core file is **invalid** and would silently corrupt every FLAG text feature.
- The node set is recoverable to **11,447 / 11,944 = 95.8%** by the reconstructed rule, and the **labeled** portion
  (8,639 nodes — the only portion that carries supervision and is scored) matches **exactly**.
- `Amazon.mat` stores no `reviewerID`, so the join must be established through the **25-dimensional feature matrix**:
  recompute Zhang et al.'s 25 features per candidate reviewer and match rows against the `.mat` features. Several
  features are near-unique integers (total helpful votes, total unhelpful votes, number of rated products, day gap),
  so exact row-matching should be highly identifying. That is the concrete next step, and it needs the sibling
  agent's `.mat` feature matrix.
- The index layout is consistent with the reconstruction: unlabeled nodes occupy `[0, 3305)` and the ambiguous band
  is exactly the unlabeled class, so the `.mat` is ordered **ambiguous-first, then labeled**.

---

## 5. Original provenance of the 25 features and the labelling rule

### 5.1 Chain of custody

```
McAuley & Leskovec 2013            raw Amazon review corpus
        |
Kumar et al., REV2, WSDM 2018      helpful-votes-as-fraud-label heuristic
        |
Zhang et al., SIGIR 2020           the 25 handcrafted user features + the >=20-votes filter + ratio rule
        |
Dou et al., CARE-GNN, CIKM 2020    Musical Instruments + 0.8/0.2 thresholds + U-P-U/U-S-U/U-V-U  ->  Amazon.mat
```

### 5.2 CARE-GNN is the graph's constructor, and it *defers* on both counts

Source: Dou, Liu, Sun, Deng, Peng, Yu, *"Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged
Fraudsters"*, CIKM 2020, arXiv:2008.08692. Retrieved from `https://ar5iv.labs.arxiv.org/html/2008.08692` (HTTP 200).

Section 4.1.1 Dataset — verbatim:

> "We use the Yelp review dataset (Rayana and Akoglu 2015) and Amazon review dataset (McAuley and Leskovec 2013) to
> study the fraudster camouflage and GNN-based fraud detection problem. [...] **The Amazon dataset includes product
> reviews under the Musical Instruments category. Similar to (Zhang et al. 2020), we label users with more than 80%
> helpful votes as benign entities and users with less than 20% helpful votes as fraudulent entities.**"

Same section, on the features — verbatim:

> "We take 32 handcrafted features from (Rayana and Akoglu 2015) (**25 handcrafted features from (Zhang et al. 2020)**
> resp.) as the raw node features for Yelp (Amazon resp.) dataset."

So the brief's quoted 80%/20% rule is **verified**, and CARE-GNN attributes **both** the features and the rule to
Zhang et al. 2020 — it is *not* the originating paper.

### 5.3 The actual originating paper: Zhang et al., SIGIR 2020

Source: Zhang, Wu, Yuan, Yin, Sheng, *"GCN-Based User Representation Learning for Unifying Robust Recommendation and
Fraudster Detection"* (GraphRfi), SIGIR 2020, arXiv:2005.10150. Retrieved from
`https://ar5iv.labs.arxiv.org/html/2005.10150` (HTTP 200).

Section 3.1, Movies & TV dataset — verbatim, and this is the key sentence:

> "This dataset contains a series of users' ratings about movies and TV crawled from Amazon by (McAuley and Leskovec
> 2013). **Following (Kumar et al. 2018), we use the helpfulness votes associated with each user's reviews for
> labelling normal users and fraudsters. Specifically, we pick users who received at least 20 votes in total. Then,
> a user is benign if the proportion of helpful votes is higher than 0.7, and fraudulent if it is below 0.3.**"

**This independently confirms the >=20-total-votes filter recovered empirically in section 4.** Zhang et al. used
0.7/0.3 on *Movies & TV*; CARE-GNN kept the >=20 filter but tightened the ratio to 0.8/0.2 and switched the category
to *Musical Instruments* — which is precisely the combination that reproduces 821/7,818/8,639.

The 25 features, enumerated verbatim from Zhang et al. Table 1 (bulleted exactly as in the source):

> "- Number of rated products
> - Length of username
> - Number and ratio of each rating level given by a user
> - **Ratio of positive and negative ratings**: The proportions of high ratings (4 and 5) and low ratings (1 and 2) of a user.
> - **Entropy of ratings**: As a measure of skewness in ratings, it can be calculated as -sum_{for all r} Pr log Pr, where Pr is the proportion that a user gives the rating of r.
> - Total number of helpful and unhelpful votes a user gets
> - The ratio and mean of helpful and unhelpful votes
> - Median, min, and max number of helpful and unhelpful votes
> - **Day gap**: The number of days between a user's first and last rating.
> - **Time entropy**: It measures the skewness in user's rating time (by year). It is calculated as -sum_{j=1}^{tau} tj log tj, where tau is the time gap between a user's first and last ratings, and tj is the proportion of ratings this user has generated in year j.
> - **Same date indicator**: It indicates whether a user's first and last comments are on the same date. The value is 1 if yes, and 0 otherwise.
> - **Median, min, max, and average of ratings**: These are further statistics to help identify fraudsters with the intuition that fraudsters are likely to give extreme ratings in order to demote/promote products.
> - **Feedback summary length**: The number of words in the feedback.
> - **Review text sentiment**: We use 1, 0, -1 to respectively represent positive, neutral and negative sentiment of a user's all reviews."

Two observations that matter for this reproduction:

1. Six of the features are functions of `helpful`/`unhelpful` vote **counts** — computable only from the 2014
   `helpful:[h,t]` pair, not from the 2018 `vote` string. Third independent confirmation of the 2014 vintage.
2. `Feedback summary length` and `Review text sentiment` are derived from **review text**, so the text FLAG needs is
   the same text the features were computed from — which is what makes feature-based row-matching viable.

### 5.4 The upstream heuristic

Zhang et al. attribute the vote-based labelling to, verbatim from their bibliography:

> "Kumar et al. (2018) Srijan Kumar, Bryan Hooi, Disha Makhija, Mohit Kumar, Christos Faloutsos, and VS Subrahmanian.
> 2018. Rev2: Fraudulent user prediction in rating platforms. In WSDM. 333-341."

**REV2 (WSDM 2018) is the ultimate origin of the labelling heuristic.** The brief's attribution of the 80/20 rule to
"the paper that introduced the benchmark" resolves to a three-paper chain, not one paper, and the 80/20 *numbers*
specifically are CARE-GNN's, not Zhang's (whose numbers were 70/30).

---

## 6. Feasibility of a reviewerID <-> graph-node mapping for FLAG

Text coverage of the reconstructed 11,447-user cohort:

```
cohort (sum_tv>=20) size: 11447
reviews/user in cohort: min=1 median=1.0 mean=4.08 max=483 total=46709
users in cohort with >=1 non-empty reviewText: 11370
users in cohort with 0 text: 77
total reviewText chars/user: min=0 median=1556 mean=4885 max=1255706
benign=7818 fraud=821 ambiguous=2808
fraud users reviews: total=1247 mean=1.52
```

- **99.3%** of cohort users (11,370 / 11,447) have at least one non-empty review; median ~1,556 characters of text
  per user. There is ample text for FLAG's per-node prompts.
- The 821 fraud users are text-poor (mean 1.52 reviews each, 1,247 reviews total). Prompt design must tolerate
  single-short-review nodes for exactly the class that matters; truncation/padding choices will disproportionately
  affect the positive class.
- 77 cohort users have **zero** text and need an explicit fallback.
- Median reviews/user is 1 — most nodes are single-review users, so "aggregate this user's review history" framings
  will degenerate for the majority of nodes.

**Assessment: feasible, but only via feature-matrix matching, and not yet proven.** The label-count agreement is
exact enough to be near-conclusive about *which corpus and which rule*, but it does not by itself establish the
*row order* of `Amazon.mat`. Recommended next step: recompute Zhang et al.'s 25 features for the 11,447 candidates
and attempt an exact row-join against the `.mat` feature matrix (sibling agent's artifact), starting with the
high-cardinality integer features. If >=95% of rows join uniquely, the mapping is established and the 497-node
residual can be characterised directly rather than inferred.

---

## 7. Unknowns

- **UNKNOWN:** origin of the 497-node shortfall (11,447 vs 11,944), entirely within the unlabeled band.
- **UNKNOWN:** exact row ordering of `Amazon.mat` — not determinable from the raw corpus alone.
- **UNKNOWN:** whether CARE-GNN used this exact SNAP snapshot (`Last-Modified: 2016-02-18`) or an earlier pull.
- **UNKNOWN:** the precise sentiment model and tokenizer behind features 24-25, so those two columns are unlikely to
  reproduce bit-exactly and should be excluded from any exact-match join key.
- **NOT VERIFIED:** CARE-GNN's published Amazon node/edge counts were not cross-read against `Amazon.mat` here;
  that is the sibling agent's scope.
