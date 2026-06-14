#!/usr/bin/env python3
"""
Build a standalone stocks-predictor executable with PyInstaller.

Usage (from project root):
  python scripts/build_predictor.py
  ./scripts/build_predictor.sh

Output:
  dist/stocks-predictor          (Linux / macOS)
  dist/stocks-predictor.exe      (Windows)
  dist/stocks-predictor-bundle/  (executable + saved_models/latest)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _venv_python() -> str:
    venv_py = ROOT / "venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    if sys.platform == "win32":
        win_py = ROOT / "venv" / "Scripts" / "python.exe"
        if win_py.is_file():
            return str(win_py)
    return sys.executable


def _install_pyinstaller(python: str) -> None:
    subprocess.check_call(
        [python, "-m", "pip", "install", "-q", "pyinstaller>=6.0"],
        cwd=ROOT,
    )


def _run_pyinstaller(python: str, onefile: bool) -> Path:
    spec = ROOT / "predictor.spec"
    cmd = [
        python,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(spec),
    ]
    if not onefile:
        # Rebuild as onedir: edit via CLI override is easier with direct pyinstaller call
        cmd = [
            python,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--name",
            "stocks-predictor",
            "--paths",
            str(ROOT),
            "--collect-submodules",
            "stable_baselines3",
            "--hidden-import",
            "data_loader",
            "--hidden-import",
            "features",
            "--hidden-import",
            "scoring",
            "--hidden-import",
            "model_store",
            "--hidden-import",
            "trading_rules",
            "--hidden-import",
            "metrics",
            "--hidden-import",
            "models.feature_columns",
            "--hidden-import",
            "hmmlearn",
            "--hidden-import",
            "gymnasium",
            "--hidden-import",
            "xgboost",
            "--hidden-import",
            "sklearn.utils._typedefs",
            "--exclude-module",
            "matplotlib",
            "--exclude-module",
            "tkinter",
            str(ROOT / "predictor.py"),
        ]
        if onefile:
            cmd.insert(4, "--onefile")
        else:
            cmd.insert(4, "--onedir")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)

    ext = ".exe" if sys.platform == "win32" else ""
    if onefile:
        return ROOT / "dist" / f"stocks-predictor{ext}"
    return ROOT / "dist" / "stocks-predictor" / f"stocks-predictor{ext}"


def _bundle_models(exe_path: Path, bundle_dir: Path) -> None:
    src = ROOT / "saved_models" / "latest"
    dst = bundle_dir / "saved_models" / "latest"
    if not (src / "metadata.json").is_file():
        print(
            "WARNING: saved_models/latest/ not found — run main.py first.\n"
            "         Copy saved_models/latest/ next to the executable manually."
        )
        return

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Bundled models → {dst}")


def _make_release_folder(exe_path: Path, onefile: bool) -> Path:
    bundle_dir = ROOT / "dist" / "stocks-predictor-bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    if onefile:
        shutil.copy2(exe_path, bundle_dir / exe_path.name)
    else:
        shutil.copytree(exe_path.parent, bundle_dir, dirs_exist_ok=True)

    _bundle_models(exe_path, bundle_dir)

    run_line = (
        f"  {exe_path.name} --strategy combined --top 20"
        if sys.platform == "win32"
        else f"  ./{exe_path.name} --strategy combined --top 20"
    )
    readme = bundle_dir / "README.txt"
    readme.write_text(
        "stocks-predictor bundle\n"
        "=======================\n\n"
        "Run (from this folder):\n"
        f"{run_line}\n\n"
        "Uses saved_models/latest/ in this folder.\n"
        "Fetches latest BVMT quotes from ilboursa.com unless --skip-refresh.\n",
        encoding="utf-8",
    )
    return bundle_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build stocks-predictor executable")
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="Build a folder dist/stocks-predictor/ instead of a single file (faster startup)",
    )
    args = parser.parse_args()

    if not (ROOT / "predictor.py").is_file():
        print(f"predictor.py not found under {ROOT}", file=sys.stderr)
        return 1

    python = _venv_python()
    print(f"Using Python: {python}")
    _install_pyinstaller(python)

    onefile = not args.onedir
    exe_path = _run_pyinstaller(python, onefile=onefile)
    if not exe_path.is_file():
        print(f"Build failed: missing {exe_path}", file=sys.stderr)
        return 1

    bundle_dir = _make_release_folder(exe_path, onefile)

    print()
    print("=" * 56)
    print(" BUILD OK")
    print("=" * 56)
    print(f"  Executable : {exe_path}")
    print(f"  Release    : {bundle_dir}/")
    print()
    print("  Run:")
    print(f"    cd {bundle_dir}")
    print(f"    ./{exe_path.name} --strategy combined --top 20")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
