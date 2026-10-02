#!/usr/bin/env python3
"""WCAG contrast check of palette.json (light+dark) and a no-hex-in-page check."""
import json, re, sys
def lum(h):
    c = [int(h[i:i+2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= .03928 else ((x + .055) / 1.055) ** 2.4 for x in c]
    return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]
def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + .05) / (lb + .05)
NEED = {"fg": 4.5, "muted": 4.5, "accent": 4.5, "ok": 4.5, "warn": 4.5, "bad": 4.5}
p = json.load(open("palette.json")); bad = []
for mode, c in p.items():
    for k, need in NEED.items():
        r = ratio(c[k], c["bg"])
        if r < need: bad.append(f"{mode}.{k} {r:.2f} < {need}")
if re.search(r"#[0-9a-fA-F]{3,6}\b", open("index.html").read()): bad.append("index.html has a hex colour; use palette vars")
print("OK" if not bad else "FAIL\n" + "\n".join(bad)); sys.exit(bool(bad))
