# Action Coder - Automated Facial Action Annotation Tool

This application automatically annotates facial action videos using speech recognition. It processes videos alongside OpenFace CSV files and generates coded annotations based on spoken commands during video recording.

## What Does This Do?

**Input:**
- Video of a patient performing facial actions with verbal cues (e.g., saying "raise eyebrows", "smile", etc.)
- One or two OpenFace CSV files with facial action unit measurements

**Output:**
- Coded CSV files with action labels for each frame
- Annotated video showing action labels overlaid on frames
- Files saved in `S2O Coded Files` directory

## Quick Start Guide

### Step 1: Install Python

You need Python 3.8 or newer. Check if you have it:

```bash
python3 --version
```

If you don't have Python, download it from [python.org](https://python.org/downloads/)

### Step 2: Install FFmpeg

FFmpeg is required for audio processing:

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
- Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Extract and add to your system PATH
- Restart Terminal/Command Prompt after installation

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

### Step 3: Install Python Dependencies

```bash
# Navigate to the S2 Action Coder folder
cd "path/to/S2 Action Coder"

# Install required packages
pip install -r requirements.txt
```

**Note:** The first run will download Whisper speech recognition models (~1-2 GB).

### Step 4: Run the Application

```bash
python main.py
```

## How to Use

### 1. File Selection
When you start the application:
- Select one or more video files (or a directory containing videos)
- The application will automatically find matching OpenFace CSV files
- Videos with "_annotated" or "_coded" suffixes are automatically excluded

### 2. Automatic Processing
The application will:
1. Load the video and CSV files
2. Extract and analyze the audio using speech recognition
3. Automatically detect spoken action commands (please note that the first time you run this code, it must download the appropriate Whisper library. This takes at least a few minutes.)
4. Create action code ranges based on detected speech
5. Pause for confirmation when commands are detected

### 3. Confirmation Prompts
When a spoken command is detected:
- **Playback pauses** at the action start
- **Audio snippet plays** of the detected speech
- **Action display shows** "Confirm: [phrase]?"
- **Click the correct action button** to confirm or reassign
- **Click "Discard"** to ignore false detections

### 4. Manual Editing
You can also manually annotate:
- **Click an action button** to select it (highlights in orange)
- **Drag on the timeline** to create a new range
- **Drag range edges** to adjust timing
- **Click a range** to select and modify it
- **Press Delete** or click "Delete Selected" to remove ranges

### 5. Save Results
- Click **"Save & Next"** to save the current file
- Files are saved to **`S2O Coded Files`** directory
- Application automatically proceeds to the next video in batch

## Supported Action Codes

| Code | Action | Speech Commands |
|------|--------|----------------|
| **RE** | Raise Eyebrows | "raise eyebrows", "eyebrows up" |
| **SS** | Soft Smile | "soft smile", "gentle smile" |
| **BS** | Big Smile | "big smile", "smile" |
| **BL** | Blink | "blink" |
| **ES** | Close Eyes Softly | "close eyes soft", "eyes closed gently" |
| **ET** | Close Eyes Tightly | "close eyes tight", "squeeze eyes" |
| **WN** | Wrinkle Nose | "wrinkle nose", "scrunch nose" |
| **PL** | Pucker Lips | "pucker lips", "kiss" |
| **LT** | Lip Together | "lips together", "press lips" |
| **SO** | Say "O" | "say o", "open mouth" |
| **SE** | Say "E" | "say e", "show teeth" |
| **BC** | Brow Cock | "cock brow", "raise one eyebrow" |
| **FR** | Frown | "frown" |
| **SN** | Snarl | "snarl", "show upper teeth" |
| **STOP** | Stop | "stop", "relax" (ends previous action) |

## Understanding the Interface

### Timeline Widget
- **Green bars** = Confirmed actions
- **Yellow bars** = Actions needing confirmation
- **Orange highlight** = Selected range
- **Red vertical line** = Current playback position
- **?? label** = Unassigned placeholder

### Action Display
- Shows current action at the playback position
- During confirmation: Shows "Confirm: [phrase]?"
- During assignment: Shows "Possible: '[text]'?"
- When creating: Shows "Pending: [action]. Drag on timeline."

### Playback Controls
- **Play/Pause** (Spacebar)
- **Frame slider** - Seek to specific frame
- **Undo/Redo** buttons for editing history

### Action Buttons
- **Number keys 1-9, 0** activate corresponding actions
- **Orange border** = Pending action for creation
- **Disabled during confirmation** prompts (except applicable actions)

## Tips for Best Results

### Recording Videos
- **Speak clearly** at the start of each action
- Use **standard command phrases** (see table above)
- **Pause briefly** between different actions
- Say **"stop"** or **"relax"** to end actions
- **Minimize background noise**

### Processing
- Review confirmation prompts carefully
- Check timeline for overlapping ranges
- Use manual editing to fine-tune timing
- Save frequently during long sessions

## Troubleshooting

### "FFmpeg not found" warning
- Ensure FFmpeg is installed and in your system PATH
- Restart Terminal/Command Prompt after installing FFmpeg
- On Mac: Try `/opt/homebrew/bin/ffmpeg` if standard install doesn't work

### Speech recognition not detecting commands
- Check audio quality in video file
- Ensure commands are spoken clearly
- Background noise may interfere with detection
- Manually create ranges if automatic detection fails

### Video or audio not playing
- Check that video file is not corrupted
- Ensure codec is supported (MP4 with H.264 recommended)
- Try converting video to MP4 format if issues persist

### Application freezes or crashes
- Close other applications to free up memory
- Check console output for error messages
- Ensure CSV files match video frame count

### Playback won't resume after confirmation
- Check console for "STUCK_GUI_DEBUG" messages
- Try clicking "Discard" to clear the prompt
- Restart the application if issue persists

## File Organization

```
S2 Action Coder/
├── main.py                      # Application entry point
├── app_controller.py            # Main application logic
├── gui_component.py             # User interface
├── qt_media_player.py           # Video playback
├── action_tracker.py            # Action range management
├── csv_handler.py               # CSV file operations
├── whisper_handler.py           # Speech recognition
├── timeline_processor.py        # Command detection logic
├── timeline_widget.py           # Interactive timeline
├── processing_manager.py        # Audio processing
├── config.py                    # Configuration and action mappings
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Output Files

After saving, you'll find in **`S2O Coded Files/`**:
- `[filename]_coded.csv` - Primary CSV with action codes
- `[filename2]_coded.csv` - Secondary CSV (if provided)
- `[filename]_coded.mp4` - Video with action labels overlaid

## Advanced Features

### Batch Processing
- Select multiple videos at startup
- Application processes them sequentially
- Progress shown in status bar
- Option to stop between files

### Keyboard Shortcuts
- **1-9, 0**: Activate action buttons
- **Spacebar**: Play/Pause
- **Delete**: Remove selected range
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo
- **Ctrl+S**: Save

### Near-Miss Detection
The application also detects speech that's *similar* to commands but doesn't match exactly. These appear as:
- Yellow "??" ranges on timeline
- Prompt asking "Possible: '[detected text]'?"
- Assign to appropriate action or discard

## Performance Notes

- **Speech recognition** runs automatically on video load
- Processing time depends on video length (typically 10-30 seconds per minute of video)
- **Whisper model** uses CPU by default (GPU acceleration available if configured)
- **Timeline editing** is disabled during audio processing

## Support

For issues or questions:
1. Check this README troubleshooting section
2. Review console output for error messages
3. Ensure all dependencies are correctly installed
4. Verify FFmpeg is accessible from command line

## Version Information

- Python 3.8+ required
- Whisper (Faster-Whisper) for speech recognition
- PyQt5 for user interface
- OpenCV for video processing
- Cross-platform (Mac, Windows, Linux)
