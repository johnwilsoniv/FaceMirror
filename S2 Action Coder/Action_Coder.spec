# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for S2 Action Coder
Builds a standalone application for video action coding with Whisper transcription

Usage:
    pyinstaller Action_Coder.spec

Output:
    dist/Action Coder.app  (macOS application bundle)

Notes:
    - FFmpeg binary must be provided separately (see README)
    - Place ffmpeg in bin/ directory before building
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all, copy_metadata
from pathlib import Path

block_cipher = None

# Determine platform
IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

# Application info
app_name = 'S2 Action Coder'
app_version = '1.0.0'

# Collect data files for dependencies
datas = []

# Copy package metadata for transformers dependencies (required for version checks)
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('huggingface_hub')
datas += copy_metadata('safetensors')
datas += copy_metadata('transformers')
datas += copy_metadata('pyyaml')
datas += copy_metadata('torch')
datas += copy_metadata('sentencepiece')
try:
    datas += copy_metadata('whisperx')
except Exception:
    print("WARNING: whisperx metadata not found")

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

# Collect PyTorch with ALL files (needed for Silero VAD and WhisperX alignment)
# CRITICAL: Must use collect_all() to include the actual Python package, not just data files
try:
    torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
    datas += torch_datas
    binaries += torch_binaries
    print(f"INFO: Collected torch - {len(torch_datas)} data files, {len(torch_binaries)} binaries")
except Exception as e:
    print(f"WARNING: Failed to collect torch: {e}")

# Collect transformers with source code (needed for WhisperX alignment Wav2Vec2)
# Must use collect_all with include_py_files to include the model source code
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')
datas += transformers_datas
binaries += transformers_binaries

# Collect whisperx with ALL files (required for word-level alignment)
# CRITICAL: Must use collect_all() to include the actual Python package
try:
    whisperx_datas, whisperx_binaries, whisperx_hiddenimports = collect_all('whisperx')
    datas += whisperx_datas
    binaries += whisperx_binaries
    print(f"INFO: Collected whisperx - {len(whisperx_datas)} data files, {len(whisperx_binaries)} binaries")
except Exception as e:
    print(f"WARNING: Failed to collect whisperx: {e}")

# Collect silero_vad (used for speech detection preprocessing)
try:
    silero_datas, silero_binaries, silero_hiddenimports = collect_all('silero_vad')
    datas += silero_datas
    binaries += silero_binaries
    print(f"INFO: Collected silero_vad - {len(silero_datas)} data files, {len(silero_binaries)} binaries")
except Exception as e:
    print(f"WARNING: Failed to collect silero_vad: {e}")

# Add patched whisperx files (fixes transformers lazy loader issue)
# These must come AFTER collect_all('whisperx') to override the originals
datas += [('patches/alignment.py', 'whisperx')]
datas += [('patches/vads/__init__.py', 'whisperx/vads')]  # Makes Pyannote optional

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
    # PyTorch for Silero VAD
    'torch',
    # Silero VAD for speech detection
    'silero_vad',
    # WhisperX for word-level alignment
    'whisperx',
    'transformers',
    'transformers.models.wav2vec2',
    'transformers.models.wav2vec2.modeling_wav2vec2',
]

# Collect PyQt5 submodules
hiddenimports += collect_submodules('PyQt5')

# Add torch hiddenimports from collect_all (needed for Silero VAD)
try:
    hiddenimports += torch_hiddenimports
except NameError:
    hiddenimports += collect_submodules('torch')

# Add transformers hiddenimports from collect_all (needed for WhisperX Wav2Vec2)
hiddenimports += transformers_hiddenimports

# Add whisperx hiddenimports from collect_all
try:
    hiddenimports += whisperx_hiddenimports
except NameError:
    try:
        hiddenimports += collect_submodules('whisperx')
    except:
        print("WARNING: whisperx submodules not found")

# Add silero_vad hiddenimports from collect_all
try:
    hiddenimports += silero_hiddenimports
except NameError:
    try:
        hiddenimports += collect_submodules('silero_vad')
    except:
        print("WARNING: silero_vad submodules not found")

# Analysis
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=['hooks/rthook-transformers.py'],
    excludes=[
        'matplotlib',
        # 'scipy',  # Required by sklearn, which is used by transformers
        'IPython',
        'jupyter',
        'notebook',
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
        bundle_identifier='com.splitface.actioncoder',
        version=app_version,
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': app_version,
            'LSMinimumSystemVersion': '10.15.0',
        },
    )
