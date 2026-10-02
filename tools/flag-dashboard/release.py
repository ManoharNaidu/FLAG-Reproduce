#!/usr/bin/env python3
"""usage: release.py release X.Y.Z "msg"  |  release.py rollback X.Y.Z   (needs git)

Scoped to this folder: it lives inside the FLAG repo, so it stages and restores only
these files and tags as dashboard-vX.Y.Z, never touching the rest of the repo."""
import datetime, os, subprocess, sys
g = lambda *a: subprocess.run(["git", *a], check=True)
def release(v, msg):
    open("VERSION", "w").write(v + "\n")
    old = open("CHANGELOG.md").read() if os.path.exists("CHANGELOG.md") else ""
    open("CHANGELOG.md", "w").write(f"## {v} ({datetime.date.today()})\n{msg}\n\n{old}")
    g("add", "-A", "--", "."); g("commit", "-qm", f"dashboard v{v}: {msg}", "--", "."); g("tag", f"dashboard-v{v}"); print("released v" + v)
def rollback(v):  # restores files from the tag as a new commit, history kept
    subprocess.run(["git", "rev-parse", "-q", "--verify", f"dashboard-v{v}"], check=True, stdout=subprocess.DEVNULL); g("checkout", f"dashboard-v{v}", "--", "."); g("commit", "-qm", f"dashboard rollback to v{v}", "--", ".")
    print(f"rolled back to v{v}; restart the service")
a = sys.argv[1:]
if len(a) == 3 and a[0] == "release": release(a[1], a[2])
elif len(a) == 2 and a[0] == "rollback": rollback(a[1])
else: sys.exit(__doc__)
