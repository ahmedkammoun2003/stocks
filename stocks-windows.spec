# -*- mode: python ; coding: utf-8 -*-
# Windows build spec — all dependencies bundled, no Python needed on target machine.

import sys
from pathlib import Path

block_cipher = None

# ── Data files to bundle ──────────────────────────────────────────────────────
# The CSV dataset is embedded so the exe is fully self-contained.
added_datas = [
    ('tunisian_stocks_30y.csv', '.'),
]

# ── Hidden imports that PyInstaller static analysis misses ────────────────────
hidden = [
    # stdlib / multiprocessing
    'multiprocessing',
    'multiprocessing.pool',
    'multiprocessing.managers',
    'concurrent.futures',
    'concurrent.futures.process',
    'threading',
    'pkg_resources',
    'setuptools',
    'pip',
    'pip._internal',
    'pip._internal.main',
]

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Exclude all heavy third-party packages to keep the EXE small and avoid 2GB limit.
        # These will be downloaded and installed natively on first run by the launcher.
        'torch', 'tensorflow', 'xgboost', 'scipy', 'numpy', 'pandas', 'matplotlib',
        'sklearn', 'stable_baselines3', 'gymnasium', 'hmmlearn', 'beautifulsoup4', 'lxml',
        'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'IPython', 'jupyter', 'notebook', 'pytest', 'black', 'flake8'
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='stocks-windows',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,           # keep True — this is a terminal trading app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='x86_64',  # 64-bit Windows
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
