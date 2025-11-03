# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for S2 Action Coder
Builds a standalone application for video action coding with Whisper transcription

Usage:
    pyinstaller Action_Coder.spec

Output:
    dist/Action Coder/  (application bundle)

Notes:
    - FFmpeg binary must be provided separately (see README)
    - Place ffmpeg in bin/ directory before building
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from pathlib import Path

block_cipher = None

# Determine platform
IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

# Application info
app_name = 'Action Coder'
app_version = '1.0.0'

# Collect data files for dependencies
datas = []

# Check for FFmpeg binary to bundle
ffmpeg_binary = None
if IS_WINDOWS:
    ffmpeg_path = Path('bin/ffmpeg.exe')
    if ffmpeg_path.exists():
        ffmpeg_binary = (str(ffmpeg_path), 'bin')
elif IS_MACOS:
    ffmpeg_path = Path('bin/ffmpeg')
    if ffmpeg_path.exists():
        ffmpeg_binary = (str(ffmpeg_path), 'bin')

binaries = []
if ffmpeg_binary:
    binaries.append(ffmpeg_binary)
    print(f"INFO: Bundling FFmpeg from {ffmpeg_binary[0]}")
else:
    print(f"WARNING: FFmpeg not found in bin/ directory. Application will require system FFmpeg.")

# Collect faster-whisper data
datas += collect_data_files('faster_whisper')

# Collect ctranslate2 data
datas += collect_data_files('ctranslate2')

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'tkinter',
    'tkinter.ttk',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.QtMultimedia',
    'PyQt5.QtMultimediaWidgets',
    'faster_whisper',
    'ctranslate2',
    'av',
    'pydub',
    'numpy',
    'pandas',
    'soundfile',
    'librosa',
    'regex',
    'tqdm',
    'huggingface_hub',
    'tokenizers',
]

# Collect PyQt5 submodules
hiddenimports += collect_submodules('PyQt5')

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
        'matplotlib',
        'scipy',
        'IPython',
        'jupyter',
        'notebook',
        # tkinter is needed for splash screen - DO NOT exclude
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
    console=False,  # GUI application, no console
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
        bundle_identifier=f'com.splitface.actioncoder',
        version=app_version,
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': app_version,
            'LSMinimumSystemVersion': '10.13.0',
        },
    )
