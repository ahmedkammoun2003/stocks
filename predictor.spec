# PyInstaller spec for stocks-predictor (run via scripts/build_predictor.py)
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH)

xgboost_binaries = collect_dynamic_libs('xgboost')
package_datas = (
    collect_data_files('xgboost')
    + collect_data_files('stable_baselines3')
    + collect_data_files('gymnasium')
)

a = Analysis(
    [str(ROOT / 'predictor.py')],
    pathex=[str(ROOT)],
    binaries=xgboost_binaries,
    datas=package_datas,
    hiddenimports=[
        'data_loader',
        'features',
        'scoring',
        'model_store',
        'metrics',
        'trading_rules',
        'models.feature_columns',
        'models.cuda_utils',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'hmmlearn',
        'hmmlearn.hmm',
        'gymnasium',
        'matplotlib',
        'matplotlib.pyplot',
        'xgboost',
        'pandas._libs.tslibs.timedeltas',
    ] + collect_submodules('stable_baselines3'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='stocks-predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
