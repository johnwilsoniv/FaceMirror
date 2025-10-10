# Split-Face Open 3 - Packaging Master Plan

**Document Version:** 1.0
**Date:** October 9, 2025
**Status:** Ready for Implementation (After Code Verification)

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Prepare Code for Bundling](#phase-1-prepare-code-for-bundling)
4. [Phase 2: Model Management](#phase-2-model-management)
5. [Phase 3: PyInstaller Configuration](#phase-3-pyinstaller-configuration)
6. [Phase 4: Build Applications](#phase-4-build-applications)
7. [Phase 5: Testing](#phase-5-testing)
8. [Phase 6: Distribution](#phase-6-distribution)
9. [Phase 7: Documentation](#phase-7-documentation)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Goal
Package three Python applications (S1 Face Mirror, S2 Action Coder, S3 Data Analysis) as standalone executables for macOS and Windows, distributed via GitHub Releases.

### Target Users
Clinician researchers with limited programming experience.

### Distribution Method
- **Platform:** GitHub Releases
- **Format:** Single package per platform containing all 3 apps
- **Size:** ~1 GB per platform (within GitHub limits)
- **License:** Apache 2.0 (matching OpenFace)

### Key Technologies
- **PyInstaller:** Bundle Python apps as executables
- **Git LFS:** Host large model files (OpenFace weights)
- **GitHub Releases:** Distribute packaged applications
- **Optional:** GitHub Pages for documentation website

---

## Prerequisites

### Software Required

**Development Machine:**
- Python 3.10+ (matching current environment)
- PyInstaller: `pip install pyinstaller`
- Git LFS: `brew install git-lfs` (macOS) or download from git-lfs.github.com
- Virtual environment with all dependencies installed

**For Testing:**
- Clean macOS VM or second Mac (for testing bundled apps)
- Windows VM or PC (for Windows builds)
- Or: Access to friend's computer for clean testing

### Repository Setup

**Initialize Git LFS:**
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3"
git lfs install
git lfs track "models/openface/*.pkl"
git lfs track "models/openface/*.pth"
git lfs track "models/openface/*.tar"
```

**Create `.gitattributes`:**
```
# Track OpenFace models with Git LFS
models/openface/*.pkl filter=lfs diff=lfs merge=lfs -text
models/openface/*.pth filter=lfs diff=lfs merge=lfs -text
models/openface/*.tar filter=lfs diff=lfs merge=lfs -text
```

---

## Phase 1: Prepare Code for Bundling

### 1.1 Path Management (All Apps)

**Create `config_paths.py` for each app:**

**S1 Face Mirror (`S1 Face Mirror/config_paths.py`):**
```python
"""Path configuration for bundled vs development environments."""
import sys
import os

APP_NAME = "1_FaceMirror_Output"
PREV_APP_NAME = None  # First in workflow

def is_bundled():
    """Check if running as PyInstaller bundle"""
    return getattr(sys, 'frozen', False)

def get_app_bundle_path():
    """Get path to app bundle resources"""
    if is_bundled():
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_output_base_dir():
    """Get base output directory for this app"""
    if is_bundled():
        home = os.path.expanduser("~")
        output_dir = os.path.join(
            home, "Documents", "SplitFace_Open3_Results", APP_NAME
        )
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_output_subdirs():
    """Get all output subdirectories"""
    base = get_output_base_dir()
    subdirs = {
        'base': base,
        'input': os.path.join(base, 'input_videos'),
        'left': os.path.join(base, 'left_mirrored'),
        'right': os.path.join(base, 'right_mirrored'),
        'openface': os.path.join(base, 'openface_results'),
    }
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
    return subdirs

def get_openface_models_dir():
    """Get OpenFace models directory"""
    if is_bundled():
        # Models bundled with app
        return os.path.join(get_app_bundle_path(), 'openface_models')
    else:
        # Development mode - use weights folder
        return os.path.join(os.path.dirname(__file__), 'weights')
```

**S2 Action Coder (`S2 Action Coder/config_paths.py`):**
```python
"""Path configuration for bundled vs development environments."""
import sys
import os

APP_NAME = "2_ActionCoder_Output"
PREV_APP_NAME = "1_FaceMirror_Output"

def is_bundled():
    return getattr(sys, 'frozen', False)

def get_default_input_dir():
    """Get default input directory (S1's OpenFace results)"""
    if is_bundled():
        home = os.path.expanduser("~")
        return os.path.join(
            home, "Documents", "SplitFace_Open3_Results",
            PREV_APP_NAME, "openface_results"
        )
    else:
        return "../S1 Face Mirror/output/openface_results"

def get_output_base_dir():
    """Get output directory for coded actions"""
    if is_bundled():
        home = os.path.expanduser("~")
        output_dir = os.path.join(
            home, "Documents", "SplitFace_Open3_Results", APP_NAME
        )
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir
```

**S3 Data Analysis (`S3 Data Analysis/config_paths.py`):**
```python
"""Path configuration for bundled vs development environments."""
import sys
import os

APP_NAME = "3_DataAnalysis_Output"
PREV_APP_NAME = "2_ActionCoder_Output"

def is_bundled():
    return getattr(sys, 'frozen', False)

def get_app_bundle_path():
    """Get path to app bundle resources"""
    if is_bundled():
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_default_input_dir():
    """Get default input directory (S2's coded actions)"""
    if is_bundled():
        home = os.path.expanduser("~")
        return os.path.join(
            home, "Documents", "SplitFace_Open3_Results", PREV_APP_NAME
        )
    else:
        # Check both possible dev locations
        option1 = "../S2 Action Coder/output"
        option2 = "../S2O Coded Files"
        return option2 if os.path.exists(option2) else option1

def get_output_base_dir():
    """Get output directory for analysis results"""
    if is_bundled():
        home = os.path.expanduser("~")
        output_dir = os.path.join(
            home, "Documents", "SplitFace_Open3_Results", APP_NAME
        )
    else:
        return "../S3O Results"

    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_models_dir():
    """Get directory containing ML models"""
    if is_bundled():
        return os.path.join(get_app_bundle_path(), 'models')
    else:
        return os.path.join(os.path.dirname(__file__), 'models')
```

### 1.2 Update Main Scripts to Use Path Functions

**S1 Face Mirror (`main.py`):**
```python
from config_paths import (
    get_output_subdirs,
    get_openface_models_dir,
    is_bundled
)

# At startup
output_dirs = get_output_subdirs()
models_dir = get_openface_models_dir()

# Update all hardcoded paths to use output_dirs dict
left_output_path = os.path.join(output_dirs['left'], f"{video_name}_left.mp4")
# etc...
```

**S2 Action Coder (`main.py`):**
```python
from config_paths import (
    get_default_input_dir,
    get_output_base_dir,
    is_bundled
)

# Set default paths in GUI
self.default_input = get_default_input_dir()
self.output_dir = get_output_base_dir()
```

**S3 Data Analysis (`main.py`):**
```python
from config_paths import (
    get_default_input_dir,
    get_output_base_dir,
    get_models_dir,
    is_bundled
)

# Set default paths
self.default_input = get_default_input_dir()
self.output_dir = get_output_base_dir()
self.models_dir = get_models_dir()
```

### 1.3 Update OpenFace Model Loading (S1)

**Modify `openface3_detector.py`:**
```python
from config_paths import get_openface_models_dir

def __init__(self, debug_mode=False, device='cpu', model_dir=None):
    if model_dir is None:
        model_dir = get_openface_models_dir()

    # Rest of initialization...
```

### 1.4 Update ML Model Loading (S3)

**Modify `paralysis_detector.py` (or wherever models are loaded):**
```python
from config_paths import get_models_dir

def load_model(zone):
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, f'{zone}_face_model.pkl')
    # Load model...
```

---

## Phase 2: Model Management

### 2.1 Organize Model Files

**Create model directories:**
```bash
mkdir -p models/openface
mkdir -p models/paralysis
```

**Copy OpenFace models:**
```bash
# Copy from current weights directory
cp "S1 Face Mirror/weights/Alignment_RetinaFace.pth" models/openface/
cp "S1 Face Mirror/weights/mobilenetV1X0.25_pretrain.tar" models/openface/

# Copy from HuggingFace cache (or download fresh)
cp ~/.cache/huggingface/hub/models--nutPace--openface_weights/blobs/b49eb62de453fbc3ef278c52b8f85630b536d0a43b82a92d6ff7a11485f12c2a models/openface/Landmark_98.pkl
cp ~/.cache/huggingface/hub/models--nutPace--openface_weights/blobs/5df84c820c9f8155ec1174252b4361bb9991271e4ffe6d0a203344eaaf95f16e models/openface/Landmark_68.pkl
cp ~/.cache/huggingface/hub/models--nutPace--openface_weights/blobs/72f45843c7a3ed9500835c507467f32e864a23971160f733aeb23797a05a5bcd models/openface/MTL_backbone.pth
```

**Copy paralysis models:**
```bash
cp "S3 Data Analysis/models/"*.pkl models/paralysis/
cp "S3 Data Analysis/models/"*.list models/paralysis/
```

### 2.2 Setup Git LFS

```bash
git lfs track "models/openface/*.pkl"
git lfs track "models/openface/*.pth"
git lfs track "models/openface/*.tar"

git add .gitattributes
git add models/
git commit -m "Add model files with Git LFS"
```

**Verify LFS:**
```bash
git lfs ls-files
# Should show all large model files
```

---

## Phase 3: PyInstaller Configuration

### 3.1 Create PyInstaller Spec Files

**Create `build/` directory:**
```bash
mkdir -p build
cd build
```

**S1 Face Mirror Spec (`build/s1_face_mirror.spec`):**
```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../S1 Face Mirror/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../models/openface/*', 'openface_models'),
        ('../S1 Face Mirror/*.py', '.'),
        ('../S1 Face Mirror/weights/*', 'weights'),  # Backup/dev mode
    ],
    hiddenimports=[
        'openface',
        'openface.face_detection',
        'openface.landmark_detection',
        'openface.Pytorch_Retinaface.layers.functions.prior_box',
        'openface.Pytorch_Retinaface.utils.box_utils',
        'torch',
        'cv2',
        'numpy',
        'PIL',
        'tqdm',
        'tkinter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # If not needed
        'scipy',  # If not needed
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
    name='FaceMirror',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
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
    name='FaceMirror',
)

app = BUNDLE(
    coll,
    name='1_FaceMirror.app',
    icon=None,  # Add icon file path if you have one
    bundle_identifier='com.splitface.facemirror',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '1.0.0',
    },
)
```

**S2 Action Coder Spec (`build/s2_action_coder.spec`):**
```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../S2 Action Coder/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../S2 Action Coder/*.py', '.'),
    ],
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtMultimedia',
        'PyQt5.QtMultimediaWidgets',
        'pandas',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='ActionCoder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name='ActionCoder',
)

app = BUNDLE(
    coll,
    name='2_ActionCoder.app',
    icon=None,
    bundle_identifier='com.splitface.actioncoder',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '1.0.0',
    },
)
```

**S3 Data Analysis Spec (`build/s3_data_analysis.spec`):**
```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../S3 Data Analysis/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../models/paralysis/*', 'models'),
        ('../S3 Data Analysis/*_features.py', '.'),
        ('../S3 Data Analysis/*.py', '.'),
    ],
    hiddenimports=[
        'sklearn',
        'sklearn.ensemble',
        'sklearn.preprocessing',
        'sklearn.calibration',
        'xgboost',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'openpyxl',
        'plotly',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='DataAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name='DataAnalysis',
)

app = BUNDLE(
    coll,
    name='3_DataAnalysis.app',
    icon=None,
    bundle_identifier='com.splitface.dataanalysis',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '1.0.0',
    },
)
```

### 3.2 Create Build Scripts

**macOS Build Script (`build/build_all_macos.sh`):**
```bash
#!/bin/bash
# Build all three applications for macOS

set -e  # Exit on error

echo "========================================"
echo "Building Split-Face Open 3 for macOS"
echo "========================================"

# Activate virtual environment
source ../.venv/bin/activate

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/

# Build S1 Face Mirror
echo ""
echo "Building S1 Face Mirror..."
pyinstaller s1_face_mirror.spec

# Build S2 Action Coder
echo ""
echo "Building S2 Action Coder..."
pyinstaller s2_action_coder.spec

# Build S3 Data Analysis
echo ""
echo "Building S3 Data Analysis..."
pyinstaller s3_data_analysis.spec

# Create distribution directory
echo ""
echo "Creating distribution package..."
mkdir -p dist/SplitFace_Open3_v1.0.0_macOS
cp -r dist/1_FaceMirror.app dist/SplitFace_Open3_v1.0.0_macOS/
cp -r dist/2_ActionCoder.app dist/SplitFace_Open3_v1.0.0_macOS/
cp -r dist/3_DataAnalysis.app dist/SplitFace_Open3_v1.0.0_macOS/
cp ../README.md dist/SplitFace_Open3_v1.0.0_macOS/README.txt
cp ../docs/UserGuide.pdf dist/SplitFace_Open3_v1.0.0_macOS/ 2>/dev/null || echo "Note: UserGuide.pdf not found, skipping"

# Create DMG (optional - requires hdiutil)
# echo "Creating DMG..."
# hdiutil create -volname "SplitFace Open 3" -srcfolder dist/SplitFace_Open3_v1.0.0_macOS -ov -format UDZO dist/SplitFace_Open3_v1.0.0_macOS.dmg

# Create ZIP
echo "Creating ZIP archive..."
cd dist
zip -r SplitFace_Open3_v1.0.0_macOS.zip SplitFace_Open3_v1.0.0_macOS/
cd ..

echo ""
echo "========================================"
echo "Build complete!"
echo "Package: build/dist/SplitFace_Open3_v1.0.0_macOS.zip"
echo "========================================"
```

**Windows Build Script (`build/build_all_windows.bat`):**
```batch
@echo off
REM Build all three applications for Windows

echo ========================================
echo Building Split-Face Open 3 for Windows
echo ========================================

REM Activate virtual environment
call ..\.venv\Scripts\activate.bat

REM Clean previous builds
echo Cleaning previous builds...
rmdir /s /q dist 2>nul
rmdir /s /q build 2>nul

REM Build S1 Face Mirror
echo.
echo Building S1 Face Mirror...
pyinstaller s1_face_mirror.spec

REM Build S2 Action Coder
echo.
echo Building S2 Action Coder...
pyinstaller s2_action_coder.spec

REM Build S3 Data Analysis
echo.
echo Building S3 Data Analysis...
pyinstaller s3_data_analysis.spec

REM Create distribution directory
echo.
echo Creating distribution package...
mkdir dist\SplitFace_Open3_v1.0.0_Windows
xcopy /E /I dist\FaceMirror dist\SplitFace_Open3_v1.0.0_Windows\1_FaceMirror
xcopy /E /I dist\ActionCoder dist\SplitFace_Open3_v1.0.0_Windows\2_ActionCoder
xcopy /E /I dist\DataAnalysis dist\SplitFace_Open3_v1.0.0_Windows\3_DataAnalysis
copy ..\README.md dist\SplitFace_Open3_v1.0.0_Windows\README.txt

REM Create ZIP
echo Creating ZIP archive...
cd dist
powershell Compress-Archive -Path SplitFace_Open3_v1.0.0_Windows -DestinationPath SplitFace_Open3_v1.0.0_Windows.zip
cd ..

echo.
echo ========================================
echo Build complete!
echo Package: build\dist\SplitFace_Open3_v1.0.0_Windows.zip
echo ========================================
```

---

## Phase 4: Build Applications

### 4.1 Test Build on Development Machine

```bash
cd build
chmod +x build_all_macos.sh
./build_all_macos.sh
```

**Expected output:**
- `dist/SplitFace_Open3_v1.0.0_macOS/` directory
- Contains 3 .app bundles
- Total size: ~1 GB

**Quick test:**
```bash
# Try running one app
open dist/SplitFace_Open3_v1.0.0_macOS/1_FaceMirror.app
```

### 4.2 Common Build Issues and Fixes

**Issue: Missing modules**
```
ModuleNotFoundError: No module named 'xxx'
```
**Fix:** Add to `hiddenimports` in spec file

**Issue: Missing data files**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/xxx'
```
**Fix:** Add to `datas` in spec file

**Issue: OpenFace models not found**
```
Error loading OpenFace models
```
**Fix:** Verify `get_openface_models_dir()` returns correct path in bundled mode

**Issue: App won't start on clean system**
- Missing system dependencies
- Check with: `otool -L dist/1_FaceMirror.app/Contents/MacOS/FaceMirror` (macOS)

---

## Phase 5: Testing

### 5.1 Test on Development Machine

**Checklist:**
- [ ] All 3 apps launch
- [ ] S1 creates output in `~/Documents/SplitFace_Open3_Results/1_FaceMirror_Output/`
- [ ] S1 processes test video successfully
- [ ] S1 runs OpenFace on mirrored videos
- [ ] S2 finds S1's output automatically
- [ ] S2 can code actions
- [ ] S3 finds S2's output automatically
- [ ] S3 can analyze coded data
- [ ] Full workflow S1→S2→S3 works end-to-end

### 5.2 Test on Clean System (Critical!)

**Setup clean macOS VM or use friend's Mac:**

**Test procedure:**
1. Copy `SplitFace_Open3_v1.0.0_macOS.zip` to clean system
2. Extract ZIP
3. Try to open each app
4. Document ALL error messages/warnings
5. Run full workflow with test data
6. Verify all outputs created correctly

**Expected issues:**
- macOS Gatekeeper warning: "App is from unidentified developer"
- **Workaround:** Right-click → Open (first time only)
- Document this in user guide!

### 5.3 Performance Testing

**Benchmark:**
- Process same video in development vs. bundled
- Compare processing times
- Should be within 5-10% of development speed

**If significantly slower:**
- Check if debug mode accidentally enabled
- Verify UPX compression not causing issues
- May need to disable UPX: `upx=False` in spec files

---

## Phase 6: Distribution

### 6.1 Prepare GitHub Repository

**Repository structure:**
```
Split-Face-Open-3/
├── README.md
├── LICENSE (Apache 2.0)
├── .gitignore
├── .gitattributes (Git LFS config)
├── src/
│   ├── s1_face_mirror/
│   ├── s2_action_coder/
│   └── s3_data_analysis/
├── models/
│   ├── openface/ (Git LFS)
│   └── paralysis/
├── build/
│   ├── *.spec
│   ├── build_all_macos.sh
│   └── build_all_windows.bat
├── docs/
│   ├── installation.md
│   ├── user_guide.md
│   └── troubleshooting.md
└── tests/
```

### 6.2 Create GitHub Release

**Steps:**

1. **Tag version:**
```bash
git tag -a v1.0.0 -m "Release v1.0.0 - Initial public release"
git push origin v1.0.0
```

2. **Create release on GitHub:**
   - Go to repository → Releases → "Draft a new release"
   - Choose tag: `v1.0.0`
   - Release title: `Split-Face Open 3 v1.0.0`

3. **Write release notes:**
```markdown
# Split-Face Open 3 v1.0.0

First public release of Split-Face Open 3 - Automated facial paralysis detection.

## What's Included

Three applications for complete facial paralysis analysis workflow:

1. **Face Mirror** - Process videos, create left/right mirrored versions
2. **Action Coder** - Manual action coding with video player
3. **Data Analysis** - Automated paralysis detection using machine learning

## Features

- Automated face detection and mirroring
- Action unit analysis using OpenFace 3.0
- Machine learning-based paralysis detection (upper, mid, lower face)
- Interactive dashboards and visualizations
- Batch processing support

## Installation

### macOS
1. Download `SplitFace_Open3_v1.0.0_macOS.zip`
2. Extract and run applications
3. First run: Right-click → Open (to bypass Gatekeeper)

### Windows
1. Download `SplitFace_Open3_v1.0.0_Windows.zip`
2. Extract to desired location
3. Run applications from extracted folder

## System Requirements

- macOS 10.14+ or Windows 10+
- 8 GB RAM recommended
- 2 GB free disk space

## Documentation

See `docs/` folder for:
- Installation guide
- User manual
- Troubleshooting

## Citation

If you use this tool in research, please cite:
[Add citation information]

## License

Apache 2.0 - See LICENSE file
```

4. **Upload files:**
   - `SplitFace_Open3_v1.0.0_macOS.zip`
   - `SplitFace_Open3_v1.0.0_Windows.zip`
   - `UserGuide.pdf` (optional)

5. **Publish release**

### 6.3 Optional: Create GitHub Pages Site

**Enable GitHub Pages:**
- Settings → Pages → Source: `gh-pages` branch or `docs/` folder
- Choose theme or use MkDocs

**Content:**
- Project overview
- Screenshots
- Download links
- Documentation
- Tutorial videos

---

## Phase 7: Documentation

### 7.1 User Guide Contents

**Installation section:**
- Download instructions
- macOS security warning workaround
- Windows Defender workaround
- First-run setup

**Workflow tutorial:**
- Step 1: Face Mirror
  - Select input videos
  - Choose output location
  - Run processing
  - OpenFace options
- Step 2: Action Coder
  - Load mirrored videos
  - Code actions
  - Save coded data
- Step 3: Data Analysis
  - Load coded data
  - Run analysis
  - View results
  - Interpret findings

**Troubleshooting:**
- "App can't be opened" (macOS)
- "Windows protected your PC" (Windows)
- "No output generated"
- "OpenFace failed"
- Performance tips

### 7.2 README.md

**Key sections:**
- What is Split-Face Open 3?
- Features
- Quick start
- System requirements
- Installation
- Usage
- Citation
- License
- Acknowledgments (OpenFace, etc.)

---

## Troubleshooting

### Build Issues

**"Permission denied" on macOS:**
```bash
chmod +x build_all_macos.sh
```

**PyInstaller not found:**
```bash
pip install pyinstaller
```

**Git LFS files not downloading:**
```bash
git lfs pull
```

**Models not found in bundle:**
- Verify `datas` paths in spec file
- Check `get_models_dir()` returns correct path
- Print debug info: `print(sys._MEIPASS)` in bundled app

### Runtime Issues

**App crashes immediately:**
- Run from terminal to see error: `/path/to/app.app/Contents/MacOS/AppName`
- Check missing dependencies

**"App is damaged" on macOS:**
- Apple Gatekeeper quarantine
- Fix: `xattr -cr /path/to/app.app`

**Antivirus false positive (Windows):**
- Document in troubleshooting
- Consider code signing (requires certificate)

### Performance Issues

**Slow processing in bundled app:**
- Disable UPX compression: `upx=False`
- Check if running in debug mode
- Verify models loaded from correct location

---

## Estimated Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Prepare code (paths, imports) | 4-6 hours |
| 2 | Organize models, Git LFS setup | 1-2 hours |
| 3 | Create PyInstaller specs | 2-3 hours |
| 4 | Build and debug first time | 3-5 hours |
| 5 | Testing on clean systems | 2-3 hours |
| 6 | GitHub release setup | 1-2 hours |
| 7 | Write documentation | 4-6 hours |
| **Total** | **First build to release** | **17-27 hours** |

**Subsequent releases:** ~2-3 hours (once pipeline established)

---

## Pre-Implementation Checklist

Before starting packaging:

- [ ] All code verified working in development
- [ ] S3 Data Analysis synkinesis cleanup complete ✅
- [ ] S1 progress GUI implemented
- [ ] Full workflow tested (S1→S2→S3) with real data
- [ ] Git repository clean and organized
- [ ] Virtual environment has all dependencies
- [ ] Test data prepared for validation

---

## Post-Release Tasks

After first release:

- [ ] Monitor GitHub issues for user feedback
- [ ] Update documentation based on user questions
- [ ] Consider adding:
  - Sample data/videos
  - Video tutorials
  - FAQ section
  - Contribution guidelines (if open source)

---

## Notes

- **No code signing initially** - too expensive ($99/year for macOS, $100+/year for Windows)
- Users will see security warnings - document workarounds
- Consider signing for v2.0 if project gains traction
- Git LFS free tier: 1 GB storage, 1 GB/month bandwidth (sufficient for small user base)
- Upgrade Git LFS if needed: $5/month for 50 GB

---

**Document maintained by:** Claude Code
**Last updated:** October 9, 2025
**Next review:** After S1 GUI implementation
