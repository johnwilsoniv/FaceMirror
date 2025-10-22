# Troubleshooting Guide - Action Coder (S2)

Common issues and solutions for Action Coder.

## Table of Contents

1. [Application Won't Start](#application-wont-start)
2. [Whisper Model Issues](#whisper-model-issues)
3. [FFmpeg Issues](#ffmpeg-issues)
4. [Video Playback Problems](#video-playback-problems)
5. [File Loading Issues](#file-loading-issues)
6. [Performance Issues](#performance-issues)
7. [Save/Export Errors](#saveexport-errors)
8. [UI/Display Issues](#uidisplay-issues)

---

## Application Won't Start

### macOS: "Action Coder is damaged and can't be opened"

**Cause**: macOS Gatekeeper blocking unsigned application

**Solution**:
```bash
xattr -cr "/Applications/Action Coder.app"
```

Or via System Preferences → Security & Privacy → General → Click "Open Anyway"

### Windows: Application closes immediately

**Cause**: Missing Visual C++ Redistributables or Python DLLs

**Solution**: Install Visual C++ Redistributable, verify Python installation

---

## Whisper Model Issues

### Model Download Stuck or Very Slow

**Symptoms**: Splash screen shows "Downloading Whisper model (3GB)..." for >15 minutes

**Solutions**:
1. Check internet connection (3GB download takes 5-10 minutes)
2. Verify disk space: Need at least 5GB free
3. Check firewall settings
4. Try from different network if corporate firewall blocks

### Manual Download (if automatic fails):

macOS/Linux:
```bash
cd ~/.cache/huggingface/hub
# Download from: https://huggingface.co/Systran/faster-whisper-large-v3
```

Windows:
```cmd
cd %LOCALAPPDATA%\huggingface\hub
# Download from: https://huggingface.co/Systran/faster-whisper-large-v3
```

---

## FFmpeg Issues

### "FFmpeg not found" Error

**macOS Solution**:
```bash
brew install ffmpeg
which ffmpeg
ffmpeg -version
```

**Windows Solution**:
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to C:fmpeg
3. Add C:fmpegin to system PATH
4. Restart terminal and application

---

## Video Playback Problems

### Video Won't Play or Shows Black Screen

**Solutions**:
1. Check video format (supported: MP4, MOV, AVI)
2. Convert unsupported formats:
```bash
ffmpeg -i input.mkv -c:v libx264 -c:a aac output.mp4
```

### Playback is Choppy or Laggy

**Solutions**:
1. Close other applications
2. Check RAM usage (<90%)
3. Reduce video resolution for 4K videos
4. Disable background apps (Time Machine, cloud sync)

### Seeking is Slow

**Solution**: Re-encode with better keyframes:
```bash
ffmpeg -i input.mp4 -g 30 -keyint_min 30 -c:v libx264 -c:a copy output.mp4
```

---

## File Loading Issues

### "Failed to load CSV files" Error

**Causes**:
1. Wrong CSV format (need 2 CSVs from Face Mirror S1)
2. File names must match: video_name_landmarks.csv and video_name_action_units.csv
3. Check file permissions

### "Video and CSV mismatch" Error

**Solution**: Verify frame counts match between video and CSV files

---

## Performance Issues

### High Memory Usage (>4GB)

**Normal Usage**: 2-3GB total (caches + Whisper model)

**If Growing Continuously**: Restart application every 2-3 hours for long sessions

### Application Freezes During Processing

**Expected**: UI remains responsive during Whisper processing

**If Frozen >5 minutes**: Force quit and check logs in logs/ directory

---

## Save/Export Errors

### "Failed to save output file" Error

**Solutions**:
1. Check write permissions on output directory
2. Verify disk space available
3. Close CSV file in Excel/viewers
4. Check file not locked by another process

### Output CSV is Empty

**Causes**:
1. No actions were coded
2. Ranges overlap or validation failed
3. Check console for errors

---

## UI/Display Issues

### Timeline Not Showing

**Solutions**:
1. Resize window to trigger redraw
2. Ensure video is loaded
3. Check status bar shows "Ready"

### Buttons Grayed Out

**Expected Behavior**: Based on app state
- No video loaded: Load video first
- Processing active: Wait for Whisper
- Prompt active: Respond to confirmation

---

## Getting More Help

1. **Check Logs**: See logs/crash_*.log files
2. **Report Issue**: Include OS version, error message, steps to reproduce
3. **GitHub Issues**: [Your Repository URL]

---

**Version**: 1.0.0
