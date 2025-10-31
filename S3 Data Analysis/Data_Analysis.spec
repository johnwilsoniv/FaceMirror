# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for S3 Data Analysis
Builds a standalone application for facial paralysis detection and analysis

Usage:
    pyinstaller Data_Analysis.spec

Output:
    dist/Data Analysis/  (application bundle)
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from pathlib import Path

block_cipher = None

# Determine platform
IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

# Application info
app_name = 'Data Analysis'
app_version = '1.0.0'

# Collect data files for dependencies
datas = []

# Add models directory (paralysis detection models)
datas += [('models', 'models')]

# Collect matplotlib data
datas += collect_data_files('matplotlib', include_py_files=False)

# Collect sklearn data
datas += collect_data_files('sklearn')

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'pandas',
    'numpy',
    'matplotlib',
    'matplotlib.backends.backend_tkagg',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.preprocessing',
    'sklearn.metrics',
    'joblib',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'cv2',
    'openpyxl',
    'xlsxwriter',
]

# Collect sklearn submodules
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('sklearn.ensemble')
hiddenimports += collect_submodules('sklearn.tree')

# Collect matplotlib backends
hiddenimports += [
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
]

# Analysis
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
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
    console=False,  # GUI application (tkinter)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Use current architecture (universal2 causes issues with Python.framework)
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
        bundle_identifier=f'com.splitface.dataanalysis',
        version=app_version,
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': app_version,
            'LSMinimumSystemVersion': '10.13.0',
        },
    )
