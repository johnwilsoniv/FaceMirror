# SplitFace Distribution Guide

Complete guide for building and distributing SplitFace applications.

## Quick Start

### Build and Package for Distribution

```bash
# Single command builds apps and creates DMG installers
./build_macos.sh
```

**What it does:**
1. Builds all three `.app` bundles with PyInstaller (~5-15 min)
2. Creates distribution-ready DMG installers (~2-5 min)

**Output:**
- `.app` bundles in each app's `dist/` folder
- Three DMG files in `DMG_Installers/`

**Note:** Currently building for ARM64 (Apple Silicon) only. Intel Mac support requires building on an Intel Mac or using a non-framework Python installation. Windows support is postponed until ARM Mac version is stable.

### Distribute to Users

Upload the DMG files to:
- GitHub Releases (recommended)
- Google Drive / Dropbox
- Your own website

Users download and install like any Mac app - no Python or technical knowledge required.

## Detailed Workflow

### Step 1: Prerequisites

**Install PyInstaller:**
```bash
conda install -c conda-forge pyinstaller
```

**Install dependencies for each app:**
```bash
cd "S1 Face Mirror" && pip install -r requirements.txt
cd "../S2 Action Coder" && pip install -r requirements.txt
cd "../S3 Data Analysis" && pip install -r requirements.txt
```

### Step 2: Build and Package

```bash
./build_macos.sh
```

This single script:
1. **Builds** ARM64 binaries (Apple Silicon) with PyInstaller
2. **Creates** DMG installers for distribution

**Output Locations:**

Application bundles:
- `S1 Face Mirror/dist/Face Mirror.app`
- `S2 Action Coder/dist/Action Coder.app`
- `S3 Data Analysis/dist/Data Analysis.app`

DMG installers in `DMG_Installers/`:
- `SplitFace-FaceMirror-v2.0.0.dmg` (~812 MB)
- `SplitFace-ActionCoder-v2.0.0.dmg` (~280 MB)
- `SplitFace-DataAnalysis-v2.0.0.dmg` (~160 MB)

**Total time:** ~7-20 minutes depending on your system

**Architecture Notes:**
- Currently builds for ARM64 (Apple Silicon) only
- Universal binary (Intel + ARM) support requires using conda or system Python instead of Python.framework
- Intel-only builds can be made on Intel Macs

### Step 3: Test Installers

Before distributing, test each DMG:

1. Double-click the DMG to mount it
2. Drag the app to Applications
3. Launch the app
4. Verify it runs without errors
5. Test core functionality

**Ideal test:** Use a Mac that has never had Python installed.

### Step 4: Distribute

#### Option A: GitHub Releases (Recommended)

1. **Create a git tag:**
   ```bash
   git tag -a v2.0.0 -m "SplitFace v2.0.0 Release"
   git push origin v2.0.0
   ```

2. **Go to GitHub:**
   - Navigate to your repository
   - Click "Releases" → "Create a new release"
   - Select the tag you created
   - Add release notes (see template below)
   - Upload the three DMG files
   - Publish release

3. **Share the release URL** with users

#### Option B: Direct File Sharing

Upload DMG files to:
- Google Drive
- Dropbox
- OneDrive
- WeTransfer
- Your own server

Share download links with users.

## Release Notes Template

```markdown
# SplitFace v2.0.0

Facial analysis suite with OpenFace 3.0 integration.

## Downloads

### macOS (Apple Silicon / ARM64)
- [S1 Face Mirror v2.0.0](link-to-dmg)
- [S2 Action Coder v2.0.0](link-to-dmg)
- [S3 Data Analysis v2.0.0](link-to-dmg)

### System Requirements
- macOS 11.0 (Big Sur) or later
- Apple Silicon (M1, M2, M3, M4) processor
- 8 GB RAM minimum (16 GB recommended for S1)
- 10 GB free disk space

**Note:** Intel Mac and Windows versions are not yet available. ARM64 version works on all Apple Silicon Macs.

### Installation
1. Download the DMG file for your desired application
2. Double-click the DMG to mount it
3. Drag the application to your Applications folder
4. Launch from Applications

No Python installation required!

### What's New in v2.0.0
- OpenFace 3.0 integration
- Native Apple Silicon (ARM64) support
- Improved memory management
- Enhanced GUI for S2
- ML-based paralysis detection in S3
- Standardized output directories
- Fixed PyInstaller bundling for all required dependencies (tkinter, matplotlib, pandas, scipy)

### Applications

**S1 Face Mirror**
Process facial videos with OpenFace 3.0 to extract Action Units and create mirrored comparisons.

**S2 Action Coder**
Interactive video coding with Whisper speech recognition for action annotation.

**S3 Data Analysis**
Automated facial paralysis detection using machine learning classifiers.

### Workflow
1. Process videos with S1 → outputs to `~/Documents/SplitFace/S1O Processed Files/`
2. Code actions with S2 → outputs to `~/Documents/SplitFace/S2O Coded Files/`
3. Analyze with S3 → outputs to `~/Documents/SplitFace/S3O Results/`

### Support
[Add contact info or issue tracker link]
```

## User Installation Instructions

Share these instructions with end users:

### Installing SplitFace Applications

1. **Download** the DMG file for your desired application
2. **Locate** the downloaded file (usually in Downloads folder)
3. **Double-click** the DMG file to mount it
4. **Drag** the application icon to the Applications folder shortcut
5. **Eject** the DMG by right-clicking and selecting "Eject"
6. **Launch** the application from your Applications folder

**First launch:** macOS may show a security warning. If so:
- Go to System Preferences → Security & Privacy
- Click "Open Anyway" for the application
- Or right-click the app → Open → confirm

### Using the Applications

Each application outputs to:
- S1: `~/Documents/SplitFace/S1O Processed Files/`
- S2: `~/Documents/SplitFace/S2O Coded Files/`
- S3: `~/Documents/SplitFace/S3O Results/`

## Updating Versions

When releasing a new version:

1. **Update version in all `config_paths.py` files:**
   ```python
   VERSION = "2.1.0"  # Update this
   ```

2. **Update version in this script:**
   Edit `create_installers.sh` and change:
   ```bash
   VERSION="2.1.0"
   ```

3. **Rebuild:**
   ```bash
   ./build_macos.sh
   ./create_installers.sh
   ```

4. **Create new release** with updated DMG files

## Technical Details

### Architecture Support
- Currently built for ARM64 (Apple Silicon) only with `target_arch=None`
- Python.framework installation does not support universal2 builds reliably
- Future universal binary support requires switching to conda or system Python
- ARM64 DMG works on M1, M2, M3, M4 Macs

### What's Bundled
Each application includes:
- Complete Python runtime
- All required Python packages
- Application code
- Resource files (models, weights)
- System libraries (Tcl/Tk for S3)

### File Sizes
| Application | DMG Size | Installed Size |
|-------------|----------|----------------|
| S1 Face Mirror | ~750 MB | ~1.5 GB |
| S2 Action Coder | ~280 MB | ~600 MB |
| S3 Data Analysis | ~160 MB | ~350 MB |

Sizes are large because they include complete Python runtimes and all dependencies.

### Security & Signing

For public distribution, consider code signing:

```bash
# Sign the app
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: Your Name" \
  "Face Mirror.app"

# Verify
codesign --verify --verbose "Face Mirror.app"
```

For macOS 10.15+, notarization is required for seamless installation.

## Windows and Intel Mac Distribution

**Status:** Postponed

Windows and Intel Mac distributions are postponed until the ARM Mac version is stable and fully tested.

**Future Plans:**
- Windows: Will use PyInstaller to create `.exe` bundles distributed as ZIP files
- Intel Mac: Will require building on an Intel Mac or switching to conda/system Python for universal2 support

**Current Focus:** Ensuring robust ARM64 (Apple Silicon) builds first

## Troubleshooting

### Build Issues

**"PyInstaller not found"**
```bash
conda install -c conda-forge pyinstaller
```

**"Module not found" during build**
```bash
cd "S1 Face Mirror"  # or S2, S3
pip install -r requirements.txt
```

**Build succeeds but app won't run**
- Test in development mode first: `python main.py`
- Check for missing resource files (weights/, models/)
- Review PyInstaller build warnings

### Distribution Issues

**DMG won't mount**
- Recreate: `./create_installers.sh`
- Check disk space
- Try on different Mac

**Users can't open app (security warning)**
- Normal on first launch
- Users should: System Preferences → Security → Open Anyway
- Or right-click app → Open

**App crashes on user's Mac**
- Verify universal binary was built (check spec files)
- Ensure all dependencies were bundled
- Check minimum macOS version compatibility

## Summary

**Your workflow:**
```bash
./build_macos.sh  # Single command: Build apps + Create DMGs (7-20 min)
# Upload to GitHub Releases
```

**User workflow:**
```
Download DMG → Mount → Drag to Applications → Launch
```

Simple, professional, and works like any other Mac application!
