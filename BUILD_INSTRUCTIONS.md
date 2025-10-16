# SplitFace v2.0.0 - Build Instructions

This document provides comprehensive instructions for building standalone applications for all three SplitFace components.

## Overview

The SplitFace suite consists of three applications:
- **S1 Face Mirror**: Facial video processing with OpenFace 3.0
- **S2 Action Coder**: Video action coding with Whisper transcription
- **S3 Data Analysis**: Facial paralysis detection and analysis

## Prerequisites

### Common Requirements (All Platforms)

1. **Python 3.10 or 3.11** (recommended)
   - Verify: `python --version` or `python3 --version`

2. **PyInstaller 6.0+**

   **If using Conda (Recommended):**
   ```bash
   conda install -c conda-forge pyinstaller
   ```

   **If using pip:**
   ```bash
   pip install pyinstaller
   ```

   **Note:** If you encounter SSL certificate errors with pip, use conda or:
   ```bash
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pyinstaller
   ```

3. **All application dependencies installed**

   **Option A: Using Conda (Recommended for building)**
   ```bash
   # Create dedicated build environment
   conda create -n splitface_build python=3.11
   conda activate splitface_build
   conda install -c conda-forge pyinstaller

   # Install app dependencies
   cd "S1 Face Mirror"
   pip install -r requirements.txt

   cd "../S2 Action Coder"
   pip install -r requirements.txt

   cd "../S3 Data Analysis"
   pip install -r requirements.txt
   ```

   **Option B: Using pip in existing environment**
   ```bash
   cd "S1 Face Mirror"
   pip install -r requirements.txt

   cd "../S2 Action Coder"
   pip install -r requirements.txt

   cd "../S3 Data Analysis"
   pip install -r requirements.txt
   ```

### Platform-Specific Requirements

#### macOS
- macOS 10.13+ (High Sierra or later)
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew (optional, for FFmpeg): `brew install ffmpeg`

#### Windows
- Windows 10 or Windows 11
- Visual C++ Redistributable (usually installed automatically)
- FFmpeg (optional, for S2): Download from https://ffmpeg.org/download.html

## Pre-Build Checklist

### S1 Face Mirror
- [ ] Verify `weights/` directory exists with OpenFace 3.0 model files
- [ ] Test that the app runs in development mode: `python main.py`

### S2 Action Coder
- [ ] (Optional) Place FFmpeg binary in `bin/` directory:
  - macOS: `bin/ffmpeg`
  - Windows: `bin/ffmpeg.exe`
- [ ] Test that the app runs in development mode: `python main.py`
- [ ] If FFmpeg not bundled, ensure it's available in system PATH

### S3 Data Analysis
- [ ] Verify `models/` directory exists with all `.pkl` model files:
  - `lower_face_model.pkl`
  - `lower_face_scaler.pkl`
  - `lower_face_features.list`
  - `mid_face_model.pkl`
  - `mid_face_scaler.pkl`
  - `mid_face_features.list`
  - `upper_face_model.pkl`
  - `upper_face_scaler.pkl`
  - `upper_face_features.list`
- [ ] Test that the app runs in development mode: `python main.py`

## Building Applications

### Option 1: Automated Build (All Apps)

#### macOS
```bash
cd "SplitFace Open3"
chmod +x build_macos.sh
./build_macos.sh
```

#### Windows
```cmd
cd "SplitFace Open3"
build_windows.bat
```

This will build all three applications sequentially.

### Option 2: Manual Build (Individual Apps)

Build each application individually:

#### S1 Face Mirror
```bash
cd "S1 Face Mirror"
pyinstaller Face_Mirror.spec
```

#### S2 Action Coder
```bash
cd "S2 Action Coder"
pyinstaller Action_Coder.spec
```

#### S3 Data Analysis
```bash
cd "S3 Data Analysis"
pyinstaller Data_Analysis.spec
```

## Build Output

### macOS
After successful build, applications will be in:
```
S1 Face Mirror/dist/Face Mirror.app
S2 Action Coder/dist/Action Coder.app
S3 Data Analysis/dist/Data Analysis.app
```

### Windows
After successful build, applications will be in:
```
S1 Face Mirror/dist/Face Mirror/Face Mirror.exe
S2 Action Coder/dist/Action Coder/Action Coder.exe
S3 Data Analysis/dist/Data Analysis/Data Analysis.exe
```

## Testing Built Applications

### S1 Face Mirror
1. Run the application
2. Select a video file for processing
3. Choose output options (OpenFace only or with mirroring)
4. Process and verify outputs in `~/Documents/SplitFace/S1O Processed Files/`

### S2 Action Coder
1. Run the application
2. Select processed videos from S1 (with matching CSV files)
3. Code actions using the interface
4. Save and verify outputs in `~/Documents/SplitFace/S2O Coded Files/`

### S3 Data Analysis
1. Run the application
2. Select data directory with coded files from S2
3. Run batch analysis
4. Verify outputs in `~/Documents/SplitFace/S3O Results/`

## Troubleshooting

### Common Issues

**"Module not found" errors during build**
- Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're in the correct environment: `which python`

**"No module named 'PyInstaller'"**
- Solution: Install PyInstaller using conda (recommended): `conda install -c conda-forge pyinstaller`
- Or with pip: `pip install pyinstaller`

**SSL Certificate Errors when installing PyInstaller**
```
SSLError(SSLCertVerificationError('"default.ssl.fastly.net" certificate name does not match...
```
- **Solution 1 (Recommended):** Use conda instead: `conda install -c conda-forge pyinstaller`
- **Solution 2:** Use trusted hosts flag:
  ```bash
  pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pyinstaller
  ```
- **Solution 3:** Update SSL certificates:
  ```bash
  conda update conda
  pip install --upgrade certifi
  pip install pyinstaller
  ```

**Build fails in conda base environment**
- Solution: Create a dedicated build environment:
  ```bash
  conda create -n splitface_build python=3.11
  conda activate splitface_build
  conda install -c conda-forge pyinstaller
  ```

**S1: "weights directory not found" error**
- Solution: Ensure `weights/` directory exists with all model files

**S2: "FFmpeg not found" warning**
- Solution: Either bundle FFmpeg in `bin/` or ensure it's in system PATH

**S3: "models directory not found" error**
- Solution: Ensure `models/` directory exists with all `.pkl` files

**macOS: "Application is damaged and can't be opened"**
- Solution: This is a Gatekeeper issue. Run:
  ```bash
  xattr -cr "Face Mirror.app"
  ```

**Windows: "The code execution cannot proceed because VCRUNTIME140.dll was not found"**
- Solution: Install Visual C++ Redistributable from Microsoft

### Build Size Issues

If the built applications are too large:

1. **Use virtual environment**: Build in a clean venv with only required packages
2. **Exclude unnecessary packages**: Edit spec files to add to `excludes` list
3. **Disable UPX compression**: Set `upx=False` in spec files if causing issues

## Distribution

### Creating Release Packages

#### macOS - Easy DMG Creation

**Option 1: All-in-One (Recommended)**
Builds apps and creates DMGs in one step:
```bash
./build_and_package_macos.sh
```

**Option 2: Create DMGs from Already-Built Apps**
If you've already built the apps:
```bash
./create_dmg.sh
```

This will create user-friendly DMG installers in `DMG_Installers/`:
- `SplitFace-FaceMirror-v2.0.0.dmg`
- `SplitFace-ActionCoder-v2.0.0.dmg`
- `SplitFace-DataAnalysis-v2.0.0.dmg`

**What Users See:**
When users double-click a DMG file, they see a window with:
- The application icon
- An Applications folder shortcut
- Arrow indicating "drag here to install"

This is the standard macOS installation experience!

#### Windows
Create a `.zip` archive:
```cmd
cd "S1 Face Mirror/dist"
tar -a -c -f "Face-Mirror-v2.0.0-Windows.zip" "Face Mirror"
```

Or use a tool like 7-Zip or WinRAR to create the archive.

## Version Updates

When updating to a new version:

1. Update `VERSION` in each `config_paths.py`
2. Update `app_version` in each `.spec` file
3. Update version in this document
4. Rebuild all applications

## Additional Notes

### Code Signing (macOS)
For distribution outside of personal use, you'll need to sign the applications:
```bash
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" "Face Mirror.app"
```

### Notarization (macOS)
For public distribution on macOS 10.15+, apps need to be notarized through Apple.

### Windows Installer
Consider creating an installer using tools like:
- Inno Setup
- NSIS
- WiX Toolset

## Support

For build issues:
1. Check the console output for specific error messages
2. Verify all prerequisites are installed
3. Test the application in development mode first
4. Review PyInstaller documentation: https://pyinstaller.org/

## License

SplitFace v2.0.0 - Research Software
