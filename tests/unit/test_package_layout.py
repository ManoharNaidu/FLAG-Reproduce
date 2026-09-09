"""Guard the package layout against module-name shadowing.

This exists because of a real bug, not as a style check.

`pyproject.toml` sets `package-dir = {"" = "src"}`, so an editable install puts
**`src/` itself on `sys.path`**. Any directory directly under `src/` therefore
becomes an importable top-level name. An early version of this repo had a
leftover empty `src/datasets/`, which Python resolved as an implicit namespace
package and which **shadowed HuggingFace's `datasets`**, breaking
`import sentence_transformers` with a misleading

    ImportError: cannot import name 'Dataset' from 'datasets' (unknown location)

The same hazard applies to `src/utils/` and `src/models/`, which would shadow
upstream FLAG's own `utils.py` and `models.py` once `methods/flag` is on the
path -- and which one won would depend on sys.path ordering.

Everything therefore lives under `src/flagbench/`. These tests keep it that way.
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

# Names that would cause a real, hard-to-diagnose collision if they ever appeared
# directly under src/. Not exhaustive -- the layout test below is the real guard.
DANGEROUS_TOP_LEVEL = {
    "datasets",    # HuggingFace datasets
    "utils",       # methods/flag/utils.py
    "models",      # methods/flag/models.py
    "test",        # methods/flag/test.py, and pytest's own discovery
    "train",       # methods/flag/train.py
    "encode",      # methods/flag/encode.py
    "chat",        # methods/flag/chat.py
    "metrics",
    "logging",
    "types",
    "json",
    "typing",
    "torch",
    "transformers",
}


def _src_entries() -> list[pathlib.Path]:
    if not SRC.exists():
        return []
    return [
        p for p in SRC.iterdir()
        if not p.name.endswith(".egg-info")
        and p.name != "__pycache__"
        and not p.name.startswith(".")
    ]


def test_src_contains_only_the_flagbench_package():
    """`src/` is on sys.path, so anything here is a top-level importable name."""
    names = sorted(p.name for p in _src_entries())
    assert names == ["flagbench"], (
        f"src/ must contain only the `flagbench` package, found {names}.\n"
        f"Anything directly under src/ becomes a top-level module name once the "
        f"package is installed editable, and can shadow a real dependency. "
        f"Move it under src/flagbench/."
    )


def test_no_dangerous_top_level_names_in_src():
    """Belt and braces: name the specific collisions we have already hit."""
    offenders = sorted(
        p.name for p in _src_entries() if p.name in DANGEROUS_TOP_LEVEL
    )
    assert not offenders, (
        f"src/{{{','.join(offenders)}}} would shadow an installed package or an "
        f"upstream FLAG module. This exact bug (src/datasets shadowing "
        f"HuggingFace datasets) has already broken this repo once."
    )


# `compat` is deliberately NOT a package. It is a directory we prepend to
# sys.path so that `import torch_scatter` resolves to our shim instead of the
# crashing wheel. Making it a package would not break that, but calling it one
# would misdescribe its role.
SYS_PATH_SHIM_DIRS = {"compat"}


def test_flagbench_subpackages_are_real_packages():
    """Namespace packages under flagbench/ would import but resolve oddly."""
    pkg = SRC / "flagbench"
    assert (pkg / "__init__.py").exists(), "flagbench is not a regular package"
    missing = [
        d.name
        for d in sorted(pkg.iterdir())
        if d.is_dir()
        and d.name != "__pycache__"
        and d.name not in SYS_PATH_SHIM_DIRS
        and not (d / "__init__.py").exists()
    ]
    assert not missing, f"subpackages without __init__.py: {missing}"


def test_compat_is_a_sys_path_shim_not_a_package():
    """Pin the intent: compat holds a top-level module we shadow by path."""
    compat = SRC / "flagbench" / "compat"
    assert compat.is_dir(), "src/flagbench/compat is missing"
    assert (compat / "torch_scatter.py").exists(), (
        "the torch_scatter shim is missing; upstream methods/flag/dga.py "
        "imports torch_scatter directly"
    )
    assert not (compat / "__init__.py").exists(), (
        "compat gained an __init__.py. It is a sys.path shim directory, not a "
        "package -- see this module's docstring and "
        "research/compatibility_notes.md section 5."
    )


def test_huggingface_datasets_is_not_shadowed():
    """The concrete regression: `datasets` must not resolve inside this repo."""
    import importlib.util

    spec = importlib.util.find_spec("datasets")
    if spec is None:
        return  # not installed at all -- fine, nothing can be shadowed
    locations = list(spec.submodule_search_locations or [])
    if spec.origin:
        locations.append(spec.origin)
    for location in locations:
        resolved = pathlib.Path(location).resolve()
        assert not resolved.is_relative_to(SRC), (
            f"`datasets` resolves to {resolved}, inside src/. It is shadowing "
            f"the real package. See this module's docstring."
        )


def test_flag_upstream_modules_are_not_shadowed_by_us():
    """`utils` / `models` must not resolve into src/ either.

    The audit tests put methods/flag on sys.path and import its `utils` and
    `models`. A same-named top-level package of ours would silently win or lose
    depending on path order.
    """
    import importlib.util

    for name in ("utils", "models"):
        spec = importlib.util.find_spec(name)
        if spec is None:
            continue
        locations = list(spec.submodule_search_locations or [])
        if spec.origin:
            locations.append(spec.origin)
        for location in locations:
            resolved = pathlib.Path(location).resolve()
            assert not resolved.is_relative_to(SRC), (
                f"`{name}` resolves to {resolved}, inside src/. It would "
                f"collide with upstream FLAG's {name}.py."
            )


def _main() -> int:
    tests = [
        (n, o) for n, o in sorted(globals().items())
        if n.startswith("test_") and callable(o)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failed.append(name)
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
