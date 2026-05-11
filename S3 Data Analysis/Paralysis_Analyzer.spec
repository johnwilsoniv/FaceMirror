# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for S3 Paralysis Analyzer
Builds a standalone application for facial paralysis detection and analysis

Usage:
    pyinstaller Paralysis_Analyzer.spec

Output:
    dist/Paralysis Analyzer.app  (macOS application bundle)
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from pathlib import Path

block_cipher = None

# Determine platform
IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

# Application info
app_name = 'S3 Data Analysis'
app_version = '1.1.1'

# Collect data files for dependencies
datas = []

# Add models directory (trained ML models)
datas += [('models', 'models')]

# Add local Python modules that are imported dynamically by joblib-loaded models
local_modules = [
    'paralysis_training_helpers.py',
    'paralysis_utils.py',
    'paralysis_config.py',
    'paralysis_detector.py',
    'paralysis_performance.py',
    'paralysis_model_trainer.py',
    'paralysis_training_pipeline.py',
    'lower_face_features.py',
    'mid_face_features.py',
    'upper_face_features.py',
    'facial_au_analyzer.py',
    'facial_au_batch_processor.py',
    'facial_au_constants.py',
    'facial_au_frame_extractor.py',
    'facial_au_gui.py',
    'facial_au_visualizer.py',
    'facial_paralysis_detection.py',
    'hardware_detection.py',
    'config_paths.py',
    'splash_screen.py',
    'training_gui.py',
    'training_summary.py',
]
for mod in local_modules:
    if Path(mod).exists():
        datas += [(mod, '.')]

# Collect sklearn data files
datas += collect_data_files('sklearn')

# Collect joblib data files
datas += collect_data_files('joblib')

# Collect xgboost data files (including VERSION file)
datas += collect_data_files('xgboost')

# Collect matplotlib data files
datas += collect_data_files('matplotlib')

# Collect imbalanced-learn data files (VERSION.txt)
datas += collect_data_files('imblearn')

# Collect seaborn data files
datas += collect_data_files('seaborn')

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'tkinter',
    'tkinter.filedialog',
    'tkinter.ttk',
    'tkinter.font',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    'PIL._tkinter_finder',
    'cv2',
    'numpy',
    'pandas',
    'joblib',
    'psutil',
    # matplotlib for visualizations
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure',
    # sklearn modules
    'sklearn',
    'sklearn.preprocessing',
    'sklearn.ensemble',
    'sklearn.svm',
    'sklearn.metrics',
    'sklearn.model_selection',
    'sklearn.utils',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    'sklearn.neighbors._partition_nodes',
    # scipy (required by sklearn)
    'scipy',
    'scipy.special',
    'scipy.sparse',
    'scipy.stats',
    # xgboost for ML models
    'xgboost',
    'xgboost.core',
    'xgboost.sklearn',
    # imbalanced-learn for SMOTE
    'imblearn',
    'imblearn.over_sampling',
    # seaborn for visualization
    'seaborn',
]
# PyObjC is macOS-only -- including these on Windows produces noisy
# "hidden import not found" warnings during PyInstaller analysis and adds
# nothing to the bundle. Append only on Darwin.
if IS_MACOS:
    hiddenimports += ['AppKit', 'Foundation', 'objc']

# Collect all sklearn submodules
hiddenimports += collect_submodules('sklearn')

# Collect all xgboost submodules. xgboost.testing imports `hypothesis`
# which is a build-time-only helper, so we install it in the build venv
# but don't ship it; the resulting hiddenimports list is filtered to drop
# the testing submodules below so they don't bloat the bundle.
_xgb_submods = collect_submodules('xgboost')
hiddenimports += [m for m in _xgb_submods if '.testing' not in m and not m.endswith('.testing')]

# Collect all imbalanced-learn submodules
hiddenimports += collect_submodules('imblearn')

# Add the XGBoost native library. Path differs by platform:
#   Windows: <site-packages>/xgboost/lib/xgboost.dll
#   macOS:   /Library/Frameworks/Python.framework/.../xgboost/lib/libxgboost.dylib
# We resolve via importlib so the path tracks whichever venv pyinstaller
# is invoked from, instead of hardcoding a brittle absolute path.
binaries = []
try:
    import importlib.util
    _xgb_spec = importlib.util.find_spec('xgboost')
    if _xgb_spec and _xgb_spec.submodule_search_locations:
        _xgb_lib_dir = Path(_xgb_spec.submodule_search_locations[0]) / 'lib'
        if IS_WINDOWS:
            _xgb_lib = _xgb_lib_dir / 'xgboost.dll'
        else:
            _xgb_lib = _xgb_lib_dir / 'libxgboost.dylib'
        if _xgb_lib.exists():
            binaries = [(str(_xgb_lib), 'xgboost/lib')]
            print(f"INFO: bundling XGBoost native lib from {_xgb_lib}")
        else:
            print(f"WARNING: XGBoost native lib not found at {_xgb_lib}")
except Exception as _e:
    print(f"WARNING: could not resolve xgboost native lib: {_e}")

# Analysis
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'jupyter',
        'notebook',
        'PyQt5',
        'PySide2',
        'torch',
        'tensorflow',
        # matplotlib IS needed for visualizations
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application with tkinter windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)

# macOS app bundle
if IS_MACOS:
    app = BUNDLE(
        coll,
        name=f'{app_name}.app',
        icon=None,
        bundle_identifier='com.splitface.paralysisanalyzer',
        version=app_version,
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': app_version,
            'LSMinimumSystemVersion': '10.15.0',
        },
    )
