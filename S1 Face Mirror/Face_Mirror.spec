# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for S1 Face Mirror
Builds a standalone application for facial video processing with OpenFace 3.0

Usage:
    pyinstaller Face_Mirror.spec

Output:
    dist/Face Mirror/  (application bundle)
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from pathlib import Path

block_cipher = None

# Determine platform
IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

# Application info
app_name = 'Face Mirror'
app_version = '2.0.0'

# Collect data files for dependencies
datas = []

# Add weights directory (OpenFace 3.0 model files)
datas += [('weights', 'weights')]

# Collect Hugging Face transformers data
datas += collect_data_files('transformers')

# Collect torch data files
datas += collect_data_files('torch')

# Collect facenet_pytorch data
try:
    datas += collect_data_files('facenet_pytorch')
except:
    pass

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'PIL._tkinter_finder',
    'cv2',
    'numpy',
    'torch',
    'torchvision',
    'transformers',
    'facenet_pytorch',
    'huggingface_hub',
    'safetensors',
    'regex',
    'requests',
    'tqdm',
    'packaging',
    'filelock',
    'typing_extensions',
]

# Collect all torch submodules
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchvision')
hiddenimports += collect_submodules('transformers')

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
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'PyQt5',
        'PySide2',
        'tkinter',
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
    console=True,  # Show console for progress output
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
        bundle_identifier=f'com.splitface.facemirror',
        version=app_version,
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': app_version,
        },
    )
