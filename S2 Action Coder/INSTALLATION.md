# Installation Guide - Action Coder (S2)

This guide covers installation for both end users and developers.

## For End Users (Recommended)

### macOS

1. **Download** the application bundle:
   - Download `Action Coder.dmg` from the releases page
   - Open the DMG file
   - Drag `Action Coder.app` to your Applications folder

2. **First Launch**:
   - Open Action Coder from Applications
   - If you see a security warning: System Preferences → Security & Privacy → Click "Open Anyway"
   - The app will download the Whisper model on first run (3GB, takes 5-10 minutes)
   - A splash screen shows download progress

3. **Verify Installation**:
   - The app should open with the main window
   - FFmpeg status should show "Ready" in the status bar
   - If FFmpeg is missing, install via Homebrew (see Troubleshooting below)

### Windows

1. **Download** the application:
   - Download `Action_Coder_Setup.exe` from the releases page
   - Run the installer
   - Follow the installation wizard

2. **First Launch**:
   - Launch Action Coder from the Start Menu
   - The app will download the Whisper model on first run (3GB)
   - Wait for the download to complete

3. **Verify Installation**:
   - Check that FFmpeg is detected (bundled with installer)
   - If issues occur, see TROUBLESHOOTING.md

## For Developers (Running from Source)

### Prerequisites

- **Python 3.10 or higher**
- **Git**
- **FFmpeg** (system installation)

### Step 1: Install System Dependencies

#### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10+
brew install python@3.10

# Install FFmpeg
brew install ffmpeg

# Verify installations
python3 --version  # Should be 3.10+
ffmpeg -version    # Should show FFmpeg version
```

#### Windows

1. **Install Python 3.10+**:
   - Download from https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **Install FFmpeg**:
   - Download from https://ffmpeg.org/download.html
   - Extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to system PATH
   - Restart terminal

3. **Verify installations**:
```cmd
python --version
ffmpeg -version
```

### Step 2: Clone Repository

```bash
git clone [Your Repository URL]
cd "S2 Action Coder"
```

### Step 3: Create Virtual Environment

#### macOS/Linux

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Windows

```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 4: Install Python Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Verify installations
pip list
```

Expected packages:
- pandas (>=1.3.0)
- numpy (>=1.20.0)
- opencv-python (>=4.5.3)
- PyQt5 (>=5.15.0)
- faster-whisper (>=0.9.0)
- thefuzz (>=0.19.0)
- ffmpeg-python (>=0.2.0)

### Step 5: Run Application

```bash
# Run from source
python main.py
```

The application will:
1. Show splash screen
2. Download Whisper model on first run (3GB, ~5-10 minutes)
3. Open main window when ready

### Step 6: Verify Configuration

```bash
# Test path configuration
python config_paths.py
```

This should print:
- Application directory
- Documents directory
- Output base directory
- Whisper cache directory
- FFmpeg path

## Building Distributable Application

### Prerequisites

```bash
pip install pyinstaller
```

### macOS Application Bundle

```bash
# Build .app bundle
pyinstaller Action_Coder.spec

# The built app will be in: dist/Action Coder.app

# Test the built application
open "dist/Action Coder.app"

# Create DMG for distribution (optional)
# [Instructions for DMG creation]
```

### Windows Executable

```cmd
# Build .exe
pyinstaller Action_Coder.spec

# The built exe will be in: dist\Action Coder\Action Coder.exe

# Test the built application
"dist\Action Coder\Action Coder.exe"

# Create installer with Inno Setup (optional)
# [Instructions for Inno Setup]
```

## Bundling FFmpeg

To bundle FFmpeg with the application:

### macOS

```bash
# Create bin directory in project root
mkdir -p bin

# Copy FFmpeg binary
cp $(which ffmpeg) bin/ffmpeg
cp $(which ffprobe) bin/ffprobe

# Update .spec file to include bin/ directory
# PyInstaller will automatically include these
```

### Windows

```cmd
# Create bin directory
mkdir bin

# Copy FFmpeg executables
copy C:\ffmpeg\bin\ffmpeg.exe bin\
copy C:\ffmpeg\bin\ffprobe.exe bin\

# Update .spec file to include bin\ directory
```

## Troubleshooting Installation

### FFmpeg Not Found

#### macOS

```bash
# Install via Homebrew
brew install ffmpeg

# Verify installation
which ffmpeg
ffmpeg -version
```

#### Windows

1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to a permanent location (e.g., `C:\ffmpeg`)
3. Add to PATH:
   - System Properties → Environment Variables
   - Edit "Path" variable
   - Add `C:\ffmpeg\bin`
   - Restart terminal

### Whisper Model Download Fails

If the automatic download fails:

1. **Manual Download**:
```bash
# Create cache directory
mkdir -p ~/.cache/huggingface/hub

# Download model manually
# Visit: https://huggingface.co/Systran/faster-whisper-large-v3
# Download all files to: ~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3
```

2. **Check Internet Connection**:
- Model download requires stable internet (3GB download)
- Use a reliable network connection
- Download may take 5-10 minutes depending on speed

3. **Check Disk Space**:
```bash
# macOS/Linux
df -h ~

# Windows
dir
```

Ensure you have at least 5GB free space.

### PyQt5 Installation Issues

#### macOS

```bash
# If PyQt5 fails to install
brew install qt@5
pip install PyQt5 --no-cache-dir
```

#### Windows

```cmd
# Use pre-compiled wheels
pip install PyQt5 --only-binary :all:
```

### faster-whisper Installation Issues

```bash
# Ensure you have compatible Python version (3.10-3.11)
python --version

# Try installing with specific version
pip install faster-whisper==0.9.0

# If CUDA errors occur (Windows with NVIDIA GPU)
pip install faster-whisper[cuda]
```

### Permission Denied Errors (macOS)

```bash
# If FFmpeg is not executable
chmod +x /opt/homebrew/bin/ffmpeg

# If application won't open
xattr -cr "/Applications/Action Coder.app"
```

## Uninstallation

### macOS

```bash
# Remove application
rm -rf "/Applications/Action Coder.app"

# Remove Whisper cache (optional, saves 3GB)
rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3

# Remove output files (optional)
# rm -rf ~/Documents/SplitFace/S2O\ Coded\ Files
```

### Windows

```cmd
# Uninstall via Control Panel → Programs and Features
# Or manually delete installation directory

# Remove Whisper cache (optional)
rmdir /s "%LOCALAPPDATA%\huggingface\hub\models--Systran--faster-whisper-large-v3"
```

## Development Setup (Advanced)

### Recommended IDE Setup

- **VS Code** with extensions:
  - Python
  - Pylance
  - PyQt5 snippets
- **PyCharm Professional** (has excellent PyQt support)

### Code Quality Tools

```bash
# Install development dependencies
pip install pyflakes flake8 black

# Run linter
flake8 *.py

# Format code
black *.py

# Check imports
pyflakes *.py
```

### Debugging

```bash
# Run with verbose logging
python main.py --verbose

# Enable profiling (for performance debugging)
# Edit main.py: Set ENABLE_PROFILING = True
python main.py
```

## Getting Help

If you encounter issues not covered here:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Search existing GitHub issues
3. Create a new issue with:
   - Your OS and version
   - Python version (`python --version`)
   - Error messages (full output)
   - Steps to reproduce

---

**Need More Help?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open a GitHub issue.
