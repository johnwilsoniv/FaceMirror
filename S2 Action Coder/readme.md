# Action Coder (S2)

**Version 1.0.0**

Action Coder is a video annotation tool for coding facial action units and behaviors from video data.

## Features

### Core Functionality
- **Video Playback**: Hardware-accelerated video playback with frame-accurate seeking
- **Action Unit Coding**: Code facial expressions and behaviors with frame-level precision
- **Speech Transcription**: Automatic audio transcription using Whisper AI (large-v3 model)
- **Timeline Editing**: Interactive timeline for creating, editing, and managing action ranges
- **Batch Processing**: Process multiple video files in sequence with automatic save
- **Undo/Redo**: Full history management with keyboard shortcuts (Ctrl+Z / Ctrl+Shift+Z)

### Advanced Features
- **Near-Miss Detection**: Automatically detects speech events near action ranges for semi-automatic coding
- **Confirmation Workflow**: Verify automatically detected ranges before finalizing
- **Range Editing**: Drag handles to adjust timing, create new ranges with mouse
- **Snippet Playback**: Play back specific ranges for review
- **CSV Export**: Exports frame-by-frame action unit data compatible with analysis tools

### Performance Optimizations
- Background frame preloader for smooth playback
- Intelligent caching (500MB RGB cache, 250MB QImage cache)
- Thread-safe video decoding
- Optimized frame extraction (11-65ms average)

## System Requirements

### Minimum Requirements
- **OS**: macOS 10.15+ or Windows 10+
- **RAM**: 8GB (16GB recommended for 4K videos)
- **Storage**: 5GB free space (3GB for Whisper model, 2GB for application)
- **CPU**: Intel Core i5 or Apple Silicon M1+
- **GPU**: Optional (CPU-only mode supported)

### Software Dependencies
- **Python**: 3.10+ (if running from source)
- **FFmpeg**: Required for audio extraction (auto-detected or bundled)
- **Internet**: Required for first-run Whisper model download (3GB)

## Quick Start

### For End Users (Pre-built Application)

1. **Download** the application bundle for your platform
2. **First Run**:
   - Launch the application
   - On first run, Whisper model will download automatically (3GB, ~5-10 minutes)
   - Subsequent launches are instant
3. **Load Video**: Click "Load Files" and select:
   - Video file (MP4, MOV, AVI)
   - Two CSV files from Face Mirror (S1) output
4. **Code Actions**: Click action buttons to create ranges, use timeline to edit
5. **Save**: Click "Save" to export coded data
6. **Batch Mode**: Load multiple file sets for sequential processing

## File Structure

### Input Files
Action Coder expects files from Face Mirror (S1) output:
```
~/Documents/SplitFace/S1O Processed Files/Combined Data/
├── video_name.mp4                    # Source video
├── video_name_landmarks.csv          # Facial landmarks (68 points)
└── video_name_action_units.csv       # Action unit intensities
```

### Output Files
Coded files are saved to:
```
~/Documents/SplitFace/S2O Coded Files/
└── video_name_coded.csv              # Frame-by-frame action codes
```

## Usage Guide

### Basic Workflow
1. **Load**: Select video and CSV files from S1 output
2. **Review**: Watch video, timeline shows Whisper-detected events
3. **Code**: Click action buttons to assign codes to ranges
4. **Edit**: Drag range handles to adjust timing
5. **Confirm**: Review auto-detected ranges (marked with "?")
6. **Save**: Export coded CSV file

### Keyboard Shortcuts
- **Space**: Play/Pause
- **Enter**: Save and load next file
- **Ctrl+Z**: Undo
- **Ctrl+Shift+Z**: Redo
- **Click Timeline**: Seek to frame

### Action Codes
- **AU Range Codes**: AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU26, AU27, AU43, AU45
- **Special Codes**:
  - **NM**: Near Miss (speech event without clear action)
  - **TBC**: To Be Coded (placeholder for manual review)

### Timeline Interaction
- **Click in axis area**: Seek to frame
- **Click on range**: Select range for editing
- **Drag range handles**: Adjust start/end frames
- **Drag in empty area**: Create new TBC range
- **Right-click range**: Delete (via selection + Delete button)

## Technical Details

### Architecture
- **GUI Framework**: PyQt5
- **Video Processing**: OpenCV (AVFoundation backend on macOS)
- **Speech Recognition**: faster-whisper (Whisper large-v3)
- **Data Processing**: pandas, numpy
- **String Matching**: thefuzz (for near-miss detection)

### Performance
- **Frame Extraction**: 11-65ms average (software decode on macOS)
- **Cache Hit Rate**: 5-7% during linear playback, 60%+ during scrubbing
- **Memory Usage**: ~2-3GB typical (includes model, caches, video buffers)
- **Whisper Processing**: ~30 seconds for 10-minute video (CPU), ~10 seconds (GPU)

### Known Limitations
- **No Hardware Acceleration on macOS**: OpenCV does not expose VideoToolbox API
- **Large Files**: Videos >1 hour may require additional memory
- **Whisper Model**: Requires 3GB disk space, downloads on first run

## Development

### Project Structure
```
S2 Action Coder/
├── main.py                      # Application entry point
├── app_controller.py            # Main application controller
├── gui_component.py             # Main window UI
├── ui_manager.py                # UI state management
├── playback_manager.py          # Video playback control
├── processing_manager.py        # Whisper processing
├── timeline_widget.py           # Timeline visualization
├── action_tracker.py            # Action range management
├── history_manager.py           # Undo/redo functionality
├── qt_media_player.py           # Video player with caching
├── whisper_handler.py           # Whisper model interface
├── timeline_processor.py        # Event timeline generation
├── batch_processor.py           # Batch file processing
├── csv_handler.py               # CSV import/export
├── config.py                    # Application configuration
├── config_paths.py              # Cross-platform path handling
└── requirements.txt             # Python dependencies
```

### Building from Source
See [INSTALLATION.md](INSTALLATION.md) for build instructions using PyInstaller.

## Citation

If you use Action Coder in your research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## License

See the main repository LICENSE file for terms of use.

## Acknowledgments

- **Whisper AI**: OpenAI's Whisper speech recognition
- **faster-whisper**: Optimized Whisper implementation by Systran
- **Face Mirror (S1)**: Facial action unit extraction tool
- **OpenCV**: Computer vision and video processing
- **PyQt5**: Cross-platform GUI framework

---

**Action Coder** is part of the SplitFace analysis toolkit for facial expression research.
