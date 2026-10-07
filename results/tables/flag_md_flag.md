> **Superseded (2026-10-07):** this file covers the earlier 64-token / 4 seeds x 2 inits run (`results/flag_md/`). The current results are in [`results/2026-10-02-flag-cosine-vs-md-main-run-report.md`](../2026-10-02-flag-cosine-vs-md-main-run-report.md). Kept for history.

# FLAG vs FLAG-MD - variant `flag`


## instagram - `cosine` vs `md_K2_matched` (mean ± std over runs)

| backbone | sampler | n | F1 | F1-macro | AUC | Precision | Recall |
|---|---|---|---|---|---|---|---|
| bwgnn | cosine | 8 | 0.1379 ± 0.0294 | 0.5300 ± 0.0078 | 0.5776 ± 0.0113 | 0.1624 ± 0.0147 | 0.1308 ± 0.0548 |
| bwgnn | md_K2_matched | 8 | 0.1511 ± 0.0129 | 0.5335 ± 0.0041 | 0.5743 ± 0.0144 | 0.1545 ± 0.0078 | 0.1510 ± 0.0280 |
| care_gnn | cosine | 8 | 0.1809 ± 0.0176 | 0.5418 ± 0.0062 | 0.6058 ± 0.0113 | 0.1659 ± 0.0162 | 0.2147 ± 0.0728 |
| care_gnn | md_K2_matched | 8 | 0.1836 ± 0.0187 | 0.5421 ± 0.0067 | 0.6037 ± 0.0136 | 0.1668 ± 0.0143 | 0.2230 ± 0.0780 |
| dga_gnn | cosine | 8 | 0.1659 ± 0.0315 | 0.5385 ± 0.0088 | 0.6030 ± 0.0084 | 0.1665 ± 0.0211 | 0.1825 ± 0.0757 |
| dga_gnn | md_K2_matched | 8 | 0.1849 ± 0.0094 | 0.5396 ± 0.0096 | 0.6073 ± 0.0083 | 0.1595 ± 0.0184 | 0.2342 ± 0.0592 |
| gat | cosine | 8 | 0.1835 ± 0.0165 | 0.5403 ± 0.0073 | 0.6060 ± 0.0188 | 0.1575 ± 0.0142 | 0.2275 ± 0.0462 |
| gat | md_K2_matched | 8 | 0.1806 ± 0.0239 | 0.5373 ± 0.0104 | 0.6033 ± 0.0137 | 0.1516 ± 0.0171 | 0.2299 ± 0.0561 |
| gcn | cosine | 8 | 0.1603 ± 0.0192 | 0.5396 ± 0.0027 | 0.6013 ± 0.0119 | 0.1743 ± 0.0211 | 0.1585 ± 0.0524 |
| gcn | md_K2_matched | 8 | 0.1759 ± 0.0172 | 0.5402 ± 0.0045 | 0.6049 ± 0.0103 | 0.1626 ± 0.0141 | 0.2039 ± 0.0609 |
| geniepath | cosine | 8 | 0.1583 ± 0.0184 | 0.5281 ± 0.0049 | 0.5687 ± 0.0135 | 0.1408 ± 0.0095 | 0.1914 ± 0.0539 |
| geniepath | md_K2_matched | 8 | 0.1422 ± 0.0286 | 0.5273 ± 0.0075 | 0.5690 ± 0.0126 | 0.1616 ± 0.0538 | 0.1501 ± 0.0531 |
| pmp | cosine | 8 | 0.1720 ± 0.0260 | 0.5422 ± 0.0084 | 0.5997 ± 0.0172 | 0.1709 ± 0.0129 | 0.1853 ± 0.0595 |
| pmp | md_K2_matched | 8 | 0.1810 ± 0.0203 | 0.5480 ± 0.0074 | 0.6019 ± 0.0131 | 0.1817 ± 0.0219 | 0.1892 ± 0.0410 |

**Paired delta (`md_K2_matched` - `cosine`), per-backbone mean over the same (seed, init) pairs:**

| metric | mean delta over backbones | backbones with delta > 0 | min | max |
|---|---|---|---|---|
| F1 | +0.0058 | 5/7 | -0.0161 | +0.0189 |
| F1-macro | +0.0011 | 5/7 | -0.0030 | +0.0058 |
| AUC | +0.0003 | 4/7 | -0.0033 | +0.0043 |
| Precision | -0.0000 | 3/7 | -0.0117 | +0.0209 |
| Recall | +0.0129 | 6/7 | -0.0413 | +0.0517 |

## reddit - `cosine` vs `md_K2_matched` (mean ± std over runs)

| backbone | sampler | n | F1 | F1-macro | AUC | Precision | Recall |
|---|---|---|---|---|---|---|---|
| bwgnn | cosine | 8 | 0.1691 ± 0.0257 | 0.5428 ± 0.0091 | 0.6190 ± 0.0178 | 0.1718 ± 0.0155 | 0.1730 ± 0.0530 |
| bwgnn | md_K2_matched | 8 | 0.1754 ± 0.0311 | 0.5448 ± 0.0097 | 0.6266 ± 0.0183 | 0.1736 ± 0.0073 | 0.1867 ± 0.0605 |
| care_gnn | cosine | 8 | 0.1667 ± 0.0246 | 0.5322 ± 0.0037 | 0.6278 ± 0.0066 | 0.1504 ± 0.0117 | 0.2067 ± 0.0762 |
| care_gnn | md_K2_matched | 8 | 0.1638 ± 0.0157 | 0.5311 ± 0.0062 | 0.6286 ± 0.0067 | 0.1462 ± 0.0088 | 0.1978 ± 0.0591 |
| dga_gnn | cosine | 8 | 0.1756 ± 0.0245 | 0.5312 ± 0.0048 | 0.6229 ± 0.0043 | 0.1474 ± 0.0092 | 0.2393 ± 0.0827 |
| dga_gnn | md_K2_matched | 8 | 0.1683 ± 0.0182 | 0.5321 ± 0.0081 | 0.6252 ± 0.0048 | 0.1476 ± 0.0118 | 0.2091 ± 0.0689 |
| gat | cosine | 8 | 0.1887 ± 0.0126 | 0.5442 ± 0.0064 | 0.6432 ± 0.0055 | 0.1648 ± 0.0154 | 0.2294 ± 0.0494 |
| gat | md_K2_matched | 8 | 0.1813 ± 0.0345 | 0.5418 ± 0.0100 | 0.6408 ± 0.0127 | 0.1639 ± 0.0102 | 0.2201 ± 0.0805 |
| gcn | cosine | 8 | 0.1408 ± 0.0197 | 0.5284 ± 0.0049 | 0.6143 ± 0.0039 | 0.1470 ± 0.0053 | 0.1402 ± 0.0370 |
| gcn | md_K2_matched | 8 | 0.1385 ± 0.0158 | 0.5269 ± 0.0033 | 0.6110 ± 0.0053 | 0.1436 ± 0.0061 | 0.1380 ± 0.0307 |
| geniepath | cosine | 8 | 0.1577 ± 0.0251 | 0.5342 ± 0.0066 | 0.6144 ± 0.0064 | 0.1556 ± 0.0096 | 0.1701 ± 0.0531 |
| geniepath | md_K2_matched | 8 | 0.1600 ± 0.0182 | 0.5334 ± 0.0052 | 0.6074 ± 0.0072 | 0.1519 ± 0.0099 | 0.1785 ± 0.0517 |
| pmp | cosine | 8 | 0.1613 ± 0.0225 | 0.5348 ± 0.0055 | 0.6287 ± 0.0085 | 0.1540 ± 0.0106 | 0.1783 ± 0.0573 |
| pmp | md_K2_matched | 8 | 0.1638 ± 0.0178 | 0.5324 ± 0.0047 | 0.6214 ± 0.0077 | 0.1460 ± 0.0077 | 0.1929 ± 0.0458 |

**Paired delta (`md_K2_matched` - `cosine`), per-backbone mean over the same (seed, init) pairs:**

| metric | mean delta over backbones | backbones with delta > 0 | min | max |
|---|---|---|---|---|
| F1 | -0.0013 | 3/7 | -0.0073 | +0.0063 |
| F1-macro | -0.0008 | 2/7 | -0.0024 | +0.0020 |
| AUC | -0.0013 | 3/7 | -0.0073 | +0.0076 |
| Precision | -0.0026 | 2/7 | -0.0079 | +0.0017 |
| Recall | -0.0020 | 3/7 | -0.0302 | +0.0146 |

## Runtime and memory (mean per run)

| sampler | train s | inference s | sampling s (one-off, per dataset) | H=Z(K)X precompute s | peak GPU MB |
|---|---|---|---|---|---|
| cosine | 66.44 | 26.14 | 11.65 | n/a | n/a |
| md_K2_matched | 64.63 | 25.95 | 21.10 | 0.20 | n/a |

_Sampling seconds for matched-budget MD include evaluating cosine to size each budget; `md_K2_top_n` is the pure MD selection cost._
