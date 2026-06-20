#!/usr/bin/env python3
"""Remove dev caches and temp dirs under the project root without touching ``.venv``."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Top-level directories to remove entirely (relative to project root)
ROOT_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".pytest_cache",
        ".pytest_tmp",
        ".ruff_cache",
        ".mypy_cache",
        ".tmp",
        ".tox",
        "htmlcov",
        "build",
        "dist",
    }
)

# Top-level file names to remove
ROOT_FILE_NAMES: frozenset[str] = frozenset({".coverage", "coverage.xml"})

# Glob patterns for removable dirs/files at project root only
ROOT_GLOBS: tuple[str, ...] = ("*.egg-info",)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _under_venv(path: Path, venv: Path) -> bool:
    try:
        path.resolve().relative_to(venv.resolve())
        return True
    except ValueError:
        return False


def _collect_pycache_dirs(root: Path, venv: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, _filename in os.walk(root, topdown=True):
        current = Path(dirpath)
        if _under_venv(current, venv):
            dirnames.clear()
            continue
        if "__pycache__" in dirnames:
            found.append(current / "__pycache__")
    return found


def _collect_pyc_files(root: Path, venv: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)
        if _under_venv(current, venv):
            continue
        if "__pycache__" in current.parts:
            continue
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                found.append(current / name)
    return found


def _remove_path(path: Path, *, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        print(f"would remove: {path}")
        return True
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"removed: {path}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete Python/build tool caches and temp files; never modifies .venv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths that would be removed without deleting.",
    )
    args = parser.parse_args(argv)

    root = _project_root()
    venv = root / ".venv"

    removed_any = False

    for name in ROOT_DIR_NAMES:
        removed_any |= _remove_path(root / name, dry_run=args.dry_run)

    for name in ROOT_FILE_NAMES:
        removed_any |= _remove_path(root / name, dry_run=args.dry_run)

    for pattern in ROOT_GLOBS:
        for path in root.glob(pattern):
            if path.is_dir() and not _under_venv(path, venv):
                removed_any |= _remove_path(path, dry_run=args.dry_run)

    for pycache in sorted(_collect_pycache_dirs(root, venv), key=lambda p: len(p.parts)):
        removed_any |= _remove_path(pycache, dry_run=args.dry_run)

    for pyc in _collect_pyc_files(root, venv):
        removed_any |= _remove_path(pyc, dry_run=args.dry_run)

    if not removed_any and not args.dry_run:
        print("Nothing to clean.")
    elif args.dry_run and not removed_any:
        print("Nothing would be removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
