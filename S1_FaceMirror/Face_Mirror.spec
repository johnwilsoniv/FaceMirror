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
app_version = '1.1.0'

# Collect data files for dependencies
datas = []

# Add weights directory (pyfaceau model files)
datas += [('weights', 'weights')]

# Add FFmpeg and FFprobe binaries (platform-specific filenames)
if IS_WINDOWS:
    ffmpeg_path = Path('bin/ffmpeg.exe')
    ffprobe_path = Path('bin/ffprobe.exe')
else:
    ffmpeg_path = Path('bin/ffmpeg')
    ffprobe_path = Path('bin/ffprobe')

if ffmpeg_path.exists():
    datas += [(str(ffmpeg_path), 'bin')]
    print(f"INFO: Bundling FFmpeg from {ffmpeg_path}")
else:
    print(f"WARNING: FFmpeg not found at {ffmpeg_path}. App will require system FFmpeg.")

if ffprobe_path.exists():
    datas += [(str(ffprobe_path), 'bin')]
    print(f"INFO: Bundling FFprobe from {ffprobe_path}")
else:
    print(f"WARNING: FFprobe not found at {ffprobe_path}. App will require system FFprobe.")

# Add local Python modules as data files (ensures they're included)
local_modules = [
    'config.py', 'config_paths.py', 'face_mirror.py', 'face_splitter.py',
    'logger.py', 'native_dialogs.py', 'openface_integration.py',
    'progress_window.py', 'pyfaceau_detector.py', 'splash_screen.py',
    'video_processor.py', 'video_rotation.py', 'au45_calculator.py',
    # performance_profiler.py is dev-only and gitignored; main.py defines
    # no-op stubs for get_profiler / set_pipeline_context inline.
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

# Collect coremltools if available (Apple Silicon acceleration only — skip on Windows)
if IS_MACOS:
    try:
        datas += collect_data_files('coremltools')
    except Exception:
        pass

# Collect onnxruntime data files (CUDA Execution Provider needs the .dll/.so set)
try:
    datas += collect_data_files('onnxruntime')
except Exception:
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
    # performance_profiler is dev-only -- see local_modules note above
    # pyfaceau stack
    'pyfaceau',
    'pyfaceau.pipeline',
    'pyfaceau.processor',
    'pyfaceau.tools',
    'pyfaceau.tools.performance_profiler',
    'pyclnf',
    'pymtcnn',
    'pyfhog',
    # ONNX Runtime providers (pymtcnn dispatches based on what's importable)
    'onnxruntime',
    'onnxruntime.capi',
    'onnxruntime.capi._pybind_state',
]

# CoreML hidden import only on macOS (coremltools is not Windows-installable)
if IS_MACOS:
    hiddenimports += ['coremltools']

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
        'pyclnf.cpp_warp',  # Exclude C extension, Python fallback will be used
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out cpp_warp binary (has Python fallback using cv2.warpAffine)
a.binaries = [b for b in a.binaries if 'cpp_warp' not in b[0]]
a.datas = [d for d in a.datas if 'cpp_warp' not in d[0] or d[0].endswith('.py')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# UPX compression: enable on macOS where it's been validated. Disable on Windows
# because UPX-packed torch / cuDNN / cuBLAS DLLs crash at runtime ("VCRUNTIME140
# load failed" and CUDA init failures are common symptoms).
USE_UPX = IS_MACOS

# target_arch: only meaningful on macOS (arm64 vs x86_64). On Windows, leaving
# this unset lets PyInstaller use the host architecture.
target_arch = 'arm64' if IS_MACOS else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=USE_UPX,
    console=False,  # GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=USE_UPX,
    upx_exclude=[
        # Belt-and-suspenders: even if upx is enabled, never compress these.
        'vcruntime*.dll', 'msvcp*.dll', 'python*.dll',
        'torch_*.dll', 'cudnn*.dll', 'cublas*.dll', 'cudart*.dll',
        'onnxruntime*.dll',
    ],
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
