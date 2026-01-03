# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for S1 Face Mirror
Builds a standalone application for facial video processing with pyfaceau

Usage:
    pyinstaller Face_Mirror.spec

Output:
    dist/Face Mirror.app  (macOS application bundle)
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from pathlib import Path

block_cipher = None

# Determine platform
IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

# Application info
app_name = 'S1 Face Mirror'
app_version = '1.0.0'

# Collect data files for dependencies
datas = []

# Add weights directory (pyfaceau model files)
datas += [('weights', 'weights')]

# Add FFmpeg binary
datas += [('bin/ffmpeg', 'bin')]

# Add local Python modules as data files (ensures they're included)
local_modules = [
    'config.py', 'config_paths.py', 'face_mirror.py', 'face_splitter.py',
    'logger.py', 'native_dialogs.py', 'openface_integration.py',
    'progress_window.py', 'pyfaceau_detector.py', 'splash_screen.py',
    'video_processor.py', 'video_rotation.py', 'au45_calculator.py',
    'performance_profiler.py'
]
for mod in local_modules:
    if Path(mod).exists():
        datas += [(mod, '.')]

# Collect pyfaceau and dependencies data files
datas += collect_data_files('pyfaceau', include_py_files=True)
datas += collect_data_files('pyclnf', include_py_files=True)
datas += collect_data_files('pymtcnn', include_py_files=True)
datas += collect_data_files('pyfhog', include_py_files=True)

# Collect torch data files
datas += collect_data_files('torch')

# Collect coremltools if available (for Apple Silicon acceleration)
try:
    datas += collect_data_files('coremltools')
except:
    pass

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'tkinter',
    'tkinter.filedialog',
    'tkinter.ttk',
    'tkinter.font',
    'tkinter.messagebox',
    'PIL._tkinter_finder',
    'cv2',
    'numpy',
    'torch',
    'scipy',
    'scipy.ndimage',
    'scipy.stats',
    'pandas',
    'tqdm',
    'psutil',
    # Local modules
    'config',
    'config_paths',
    'face_mirror',
    'face_splitter',
    'logger',
    'native_dialogs',
    'openface_integration',
    'progress_window',
    'pyfaceau_detector',
    'splash_screen',
    'video_processor',
    'video_rotation',
    'au45_calculator',
    'performance_profiler',
    # pyfaceau stack
    'pyfaceau',
    'pyfaceau.pipeline',
    'pyfaceau.processor',
    'pyfaceau.tools',
    'pyfaceau.tools.performance_profiler',
    'pyclnf',
    'pymtcnn',
    'pyfhog',
    # CoreML for Apple Silicon
    'coremltools',
]

# Collect all torch submodules
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('pyfaceau')
hiddenimports += collect_submodules('pyclnf')
hiddenimports += collect_submodules('pymtcnn')

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
        'matplotlib',  # Not needed for S1
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
        bundle_identifier='com.splitface.facemirror',
        version=app_version,
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': app_version,
            'LSMinimumSystemVersion': '10.15.0',
        },
    )
