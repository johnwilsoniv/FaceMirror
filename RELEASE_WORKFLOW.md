# SplitFace Release Workflow

This guide explains how to build and distribute SplitFace applications to end users.

## Important: Who Does What

### You (Developer) - ONE TIME BUILD
1. Build the applications
2. Create DMG installers
3. Upload DMGs to GitHub Releases or file sharing

### End Users - SIMPLE DOWNLOAD & INSTALL
1. Download the DMG file
2. Double-click and drag to Applications
3. Done! No building required

**End users NEVER need to build anything or run scripts!**

## For You: Creating Release Files

### Step 1: Build Universal Binaries

Build applications that work on **both Intel and Apple Silicon Macs**:

```bash
# From the project root
./build_macos.sh
```

This creates `.app` bundles in each app's `dist/` folder with universal binaries.

### Step 2: Create DMG Installers

Create distribution-ready DMG files:

```bash
./create_simple_dmg.sh
```

This creates three DMG files in `DMG_Installers/`:
```
DMG_Installers/
├── SplitFace-FaceMirror-v2.0.0.dmg
├── SplitFace-ActionCoder-v2.0.0.dmg
└── SplitFace-DataAnalysis-v2.0.0.dmg
```

### Step 3: Distribute to Users

**Option A: GitHub Releases (Recommended)**

1. Create a git tag and release:
   ```bash
   git tag -a v2.0.0 -m "SplitFace v2.0.0 Release"
   git push origin v2.0.0
   ```

2. Go to GitHub → Releases → Create New Release

3. Upload the three DMG files from `DMG_Installers/`

4. Users download directly from GitHub Releases

**Option B: File Sharing Service**

Upload DMG files to:
- Google Drive
- Dropbox
- OneDrive
- Your own website
- Any file hosting service

Share the download links with users.

## For End Users: Installation

**Users receive:**
- A download link to a DMG file (e.g., `SplitFace-FaceMirror-v2.0.0.dmg`)

**Installation steps:**
1. Download the DMG file
2. Double-click the DMG to mount it
3. Drag the app to the Applications folder
4. Eject the DMG
5. Open the app from Applications

**That's it!** No Python, no terminal, no building required.

## Universal Binary Benefits

With `target_arch='universal2'`, each DMG works on:
- ✅ Intel Macs (x86_64)
- ✅ Apple Silicon Macs (arm64 - M1, M2, M3, etc.)

Users don't need to know what chip they have - one DMG works for everyone.

## File Sizes (Approximate)

| Application    | DMG Size |
|----------------|----------|
| S1 Face Mirror | ~750 MB  |
| S2 Action Coder| ~280 MB  |
| S3 Data Analysis| ~160 MB |

Universal binaries are slightly larger but provide better compatibility.

## Complete Build and Release Commands

**For a new release, run these commands:**

```bash
# 1. Update version in config_paths.py files (if needed)

# 2. Build everything
./build_macos.sh

# 3. Create DMGs
./create_simple_dmg.sh

# 4. Test the DMGs (mount and run apps)
open DMG_Installers/SplitFace-FaceMirror-v2.0.0.dmg

# 5. Commit any changes
git add .
git commit -m "Release v2.0.0"
git push

# 6. Create release tag
git tag -a v2.0.0 -m "SplitFace v2.0.0"
git push origin v2.0.0

# 7. Upload DMGs to GitHub Releases page
```

## What NOT to Share

**Don't distribute:**
- ❌ The `dist/` folders (these are build artifacts)
- ❌ The `.app` bundles directly (wrap them in DMGs)
- ❌ Build scripts (users don't need these)
- ❌ Source code (unless you want developers to contribute)

**Do distribute:**
- ✅ DMG files from `DMG_Installers/`
- ✅ README or installation guide
- ✅ Release notes

## Testing Before Release

**Always test on a clean machine before public release:**

1. Mount the DMG
2. Drag app to Applications
3. Launch the app
4. Verify it runs without errors
5. Test core functionality
6. Check it doesn't ask for Python or dependencies

**Ideal test:** Use a friend's Mac that has never had Python installed.

## Updating Versions

When releasing a new version:

1. Update `VERSION` in all three `config_paths.py` files:
   ```python
   VERSION = "2.1.0"  # Update this
   ```

2. Update version in spec files if needed

3. Rebuild and recreate DMGs

4. DMG filenames will automatically include new version

## Windows Distribution

For Windows users, the process is similar:

1. Build on Windows: `build_windows.bat`
2. Zip each folder in `dist/`
3. Distribute ZIP files
4. Users extract and run `.exe`

Windows doesn't use DMG files - ZIP archives are standard.

## Common Questions

**Q: Do users need to install Python?**
A: No! The DMGs contain everything needed.

**Q: Do users need to run build scripts?**
A: No! You build once, they just download and install.

**Q: Can I share the .app directly?**
A: Technically yes, but DMGs provide a better installation experience.

**Q: Why are DMGs so large?**
A: They contain the entire Python runtime, all dependencies, and ML models. This is intentional - users get a complete, working application.

**Q: What's the difference between the DMG and the .app?**
A: The DMG is an installer that contains the .app. Users mount the DMG to extract and install the .app.

## Summary

**Your workflow:**
```bash
./build_macos.sh        # Build apps
./create_simple_dmg.sh  # Create DMGs
# Upload DMGs to GitHub Releases
```

**User workflow:**
```
1. Download DMG
2. Drag to Applications
3. Launch app
```

The DMGs in `DMG_Installers/` are the final distribution files!
