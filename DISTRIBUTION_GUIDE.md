# SplitFace Distribution Guide

Guide for creating professional installers for public distribution.

## macOS Distribution (DMG Files)

### Quick Start

**Build everything in one command:**
```bash
./build_and_package_macos.sh
```

This creates three DMG installers in the `DMG_Installers/` folder, ready to distribute.

### What Users Experience

1. **Download** the DMG file (e.g., `SplitFace-FaceMirror-v2.0.0.dmg`)
2. **Double-click** the DMG file
3. **Drag** the app icon to the Applications folder shortcut
4. **Done!** The app is installed

### DMG Installer Contents

Each DMG provides:
- ✓ Beautiful drag-and-drop interface
- ✓ Applications folder shortcut
- ✓ Self-contained app bundle (no Python required)
- ✓ Professional appearance

### Distribution Files Created

After running `./build_and_package_macos.sh`:
```
DMG_Installers/
├── SplitFace-FaceMirror-v2.0.0.dmg    (~200-500 MB)
├── SplitFace-ActionCoder-v2.0.0.dmg   (~150-300 MB)
└── SplitFace-DataAnalysis-v2.0.0.dmg  (~100-200 MB)
```

Upload these DMG files to:
- GitHub Releases
- File sharing service
- Your website
- Distribution platform

## Windows Distribution (ZIP Files)

### Creating Windows Installers

After building with `build_windows.bat`, create ZIP archives:

```cmd
cd "S1 Face Mirror/dist"
tar -a -c -f "SplitFace-FaceMirror-v2.0.0-Windows.zip" "Face Mirror"

cd "../../S2 Action Coder/dist"
tar -a -c -f "SplitFace-ActionCoder-v2.0.0-Windows.zip" "Action Coder"

cd "../../S3 Data Analysis/dist"
tar -a -c -f "SplitFace-DataAnalysis-v2.0.0-Windows.zip" "Data Analysis"
```

Or use 7-Zip/WinRAR to create ZIP files from each dist folder.

### What Users Experience

1. **Download** the ZIP file
2. **Extract** all files to a folder
3. **Run** the .exe file
4. **Optional:** Create a desktop shortcut

### Distribution Files

```
SplitFace-FaceMirror-v2.0.0-Windows.zip
SplitFace-ActionCoder-v2.0.0-Windows.zip
SplitFace-DataAnalysis-v2.0.0-Windows.zip
```

## File Sizes (Approximate)

| Application    | macOS DMG  | Windows ZIP |
|----------------|------------|-------------|
| S1 Face Mirror | 400 MB     | 350 MB      |
| S2 Action Coder| 250 MB     | 200 MB      |
| S3 Data Analysis| 150 MB    | 120 MB      |

*Sizes vary based on dependencies and PyInstaller compression*

## Advanced: Creating Windows Installer (Optional)

For a more professional Windows experience, create an installer using:

### Inno Setup (Recommended)

1. Install Inno Setup: https://jrsoftware.org/isinfo.php
2. Create a script for each app (example for S1):

```iss
[Setup]
AppName=SplitFace Face Mirror
AppVersion=2.0.0
DefaultDirName={autopf}\SplitFace\Face Mirror
DefaultGroupName=SplitFace
OutputDir=Installers
OutputBaseFilename=SplitFace-FaceMirror-v2.0.0-Setup

[Files]
Source: "S1 Face Mirror\dist\Face Mirror\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\Face Mirror"; Filename: "{app}\Face Mirror.exe"
Name: "{autodesktop}\Face Mirror"; Filename: "{app}\Face Mirror.exe"
```

This creates a professional Windows installer with:
- Setup wizard
- Installation directory selection
- Start menu shortcuts
- Desktop shortcut option
- Uninstaller

## Publishing to GitHub Releases

### 1. Create a Release

```bash
git tag -a v2.0.0 -m "SplitFace v2.0.0 - OpenFace 3.0 Release"
git push origin v2.0.0
```

### 2. Upload Distribution Files

Go to GitHub → Releases → Create New Release

**Release Title:** SplitFace v2.0.0

**Release Notes Example:**
```markdown
## SplitFace v2.0.0

Major update featuring OpenFace 3.0 integration and cross-platform support.

### Downloads

#### macOS (10.13+)
- [Face Mirror v2.0.0 (DMG)](SplitFace-FaceMirror-v2.0.0.dmg)
- [Action Coder v2.0.0 (DMG)](SplitFace-ActionCoder-v2.0.0.dmg)
- [Data Analysis v2.0.0 (DMG)](SplitFace-DataAnalysis-v2.0.0.dmg)

#### Windows (10/11)
- [Face Mirror v2.0.0 (ZIP)](SplitFace-FaceMirror-v2.0.0-Windows.zip)
- [Action Coder v2.0.0 (ZIP)](SplitFace-ActionCoder-v2.0.0-Windows.zip)
- [Data Analysis v2.0.0 (ZIP)](SplitFace-DataAnalysis-v2.0.0-Windows.zip)

### Installation

**macOS:** Double-click DMG, drag to Applications
**Windows:** Extract ZIP, run .exe file

### What's New
- OpenFace 3.0 integration
- Improved memory management
- Cross-platform builds
- Standardized output directories
[... more details ...]
```

## Code Signing (Production)

For public distribution, code signing is recommended:

### macOS Code Signing

```bash
# Sign the app
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: Your Name" \
  "Face Mirror.app"

# Verify signature
codesign --verify --verbose "Face Mirror.app"
spctl --assess --verbose "Face Mirror.app"

# Notarize (required for macOS 10.15+)
# See: https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution
```

### Windows Code Signing

Requires a code signing certificate from a trusted CA:
- DigiCert
- Sectigo
- GlobalSign

```cmd
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com "Face Mirror.exe"
```

## Distribution Checklist

Before releasing:

- [ ] Test on clean macOS system (no Python installed)
- [ ] Test on clean Windows system (no Python installed)
- [ ] Verify all dependencies are bundled
- [ ] Test all major features work
- [ ] Create release notes
- [ ] Build DMG files (macOS)
- [ ] Create ZIP archives (Windows)
- [ ] Upload to GitHub Releases
- [ ] Update documentation links
- [ ] Announce release

## Support Documentation

Provide users with:
1. Installation instructions (included in release notes)
2. Quick start guide
3. System requirements
4. Troubleshooting guide
5. Contact information for support

## License and Attribution

Ensure distribution files include:
- LICENSE file
- Third-party licenses (for bundled dependencies)
- Attribution for OpenFace, PyTorch, etc.

## Questions?

See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for building details.
