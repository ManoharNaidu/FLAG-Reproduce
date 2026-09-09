"""Extract FLAG's prompt strings verbatim from the official source via AST.

We parse rather than import because chat.py/chat1.py execute torch.load at module
scope. AST extraction guarantees the prompts are byte-identical to upstream.

Usage: python -m scripts.preprocess.extract_prompts
"""
import ast
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLAG = ROOT / "methods" / "flag"
OUT = ROOT / "prompts"

# (source file, dataset dir) -- chat.py is Reddit, chat1.py is Instagram.
SOURCES = [("chat.py", "reddit"), ("chat1.py", "instagram")]

# FLAG's internal name -> the paper's terminology.
NAME_MAP = {
    "system_instruction": "system_instruction",
    "global_prompt": "global",
    "unique_prompt": "discriminative",   # paper: discriminative text
    "common_prompt": "residual",         # paper: residual text
}


def extract(path):
    """Return {assigned_name: str_value} for module-level string assignments."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in NAME_MAP:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        if isinstance(value, str):
            found[target.id] = value
    return found


def main():
    manifest = {
        "source_repo": "https://github.com/BUPT-GAMMA/FLAG",
        "source_commit": "cb83944ed8a8a9b070a3f5a167d363973369fc80",
        "prompt_version": "flag-official-v1",
        "extraction": "ast.literal_eval of module-level assignments",
        "prompts": {},
    }

    for filename, dataset in SOURCES:
        src = FLAG / filename
        if not src.exists():
            raise SystemExit(
                f"missing {src}\nRun scripts/setup/fetch_methods.sh first."
            )
        outdir = OUT / dataset
        outdir.mkdir(parents=True, exist_ok=True)
        for var, text in extract(src).items():
            role = NAME_MAP[var]
            dest = outdir / f"{role}.txt"
            dest.write_text(text, encoding="utf-8", newline="\n")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            manifest["prompts"][f"{dataset}/{role}"] = {
                "source_file": filename,
                "source_symbol": var,
                "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
                "chars": len(text),
                "sha256": digest,
            }
            print(f"{dataset:10s} {role:20s} {len(text):5d} chars  {digest[:16]}")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'manifest.json'} ({len(manifest['prompts'])} prompts)")


if __name__ == "__main__":
    main()
