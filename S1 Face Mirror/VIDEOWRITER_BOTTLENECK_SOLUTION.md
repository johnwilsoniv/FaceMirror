# VideoWriter Bottleneck - Problem & Solution

## 📋 PROBLEM SUMMARY

**Current Bottleneck:** `cv2.VideoWriter` is extremely slow and blocking batch processing transitions.

### Performance Metrics:
- **cv2.VideoWriter speed**: 46ms per frame (~22 fps max)
- **mp4v codec**: 1.7s per 100 frames
- **H.264 (avc1) codec**: 1.4s per 100 frames (only 18% faster)
- **Impact**: 3-second delays between batches, 40% processing slowdown when queuing

### Current Location:
**File:** `/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/video_processor.py`
**Lines:** 144-147 (VideoWriter initialization), 280-281 (frame writing)

```python
# Current slow approach
fourcc = cv2.VideoWriter_fourcc(*'avc1')
right_writer = cv2.VideoWriter(str(anatomical_right_output), fourcc, fps, (width, height))
# ...
right_writer.write(right_face.astype(np.uint8))
```

---

## ✅ RECOMMENDED SOLUTION: Direct FFmpeg Piping

### Why FFmpeg is Better:
1. **Hardware acceleration** - VideoToolbox on macOS, NVENC on NVIDIA, QSV on Intel
2. **3-10x faster** than cv2.VideoWriter
3. **Better quality** - more codec options and tuning
4. **Non-blocking** - can write to pipe without blocking
5. **Already available** - ffmpeg is already used for video rotation

### Implementation Approach:

#### Option 1: FFmpeg Stdin Pipe (RECOMMENDED)
Write raw frames directly to ffmpeg's stdin, let it encode in real-time.

**Pseudocode:**
```python
import subprocess
import numpy as np

# Start ffmpeg process with pipe input
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # Overwrite output
    '-f', 'rawvideo',  # Input format
    '-vcodec', 'rawvideo',
    '-s', f'{width}x{height}',  # Size
    '-pix_fmt', 'bgr24',  # OpenCV default
    '-r', str(fps),  # Frame rate
    '-i', '-',  # Read from stdin
    '-an',  # No audio
    '-vcodec', 'h264_videotoolbox',  # Hardware encoder (macOS)
    # OR: '-vcodec', 'h264_nvenc',  # For NVIDIA
    # OR: '-vcodec', 'libx264', '-preset', 'ultrafast',  # CPU fallback
    '-pix_fmt', 'yuv420p',
    '-crf', '18',  # Quality (lower = better)
    str(output_path)
]

process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

# Write frames (fast - just dumps to pipe)
for frame in frames:
    process.stdin.write(frame.tobytes())

# Finish
process.stdin.close()
process.wait()
```

**Expected Performance:**
- **Pipe write**: <1ms per frame (nearly instant)
- **Encoding**: Happens in background, hardware-accelerated
- **Total overhead**: Minimal, mostly I/O bound

---

#### Option 2: imageio-ffmpeg Library
Higher-level wrapper, easier to use but slightly less control.

```python
import imageio

writer = imageio.get_writer(output_path, fps=fps, codec='h264_videotoolbox',
                             quality=8, pixelformat='yuv420p')
for frame in frames:
    writer.append_data(frame)
writer.close()
```

---

#### Option 3: PyAV (Most Pythonic)
Best for fine-grained control over encoding parameters.

```python
import av

container = av.open(output_path, mode='w')
stream = container.add_stream('h264', rate=fps)
stream.width = width
stream.height = height
stream.pix_fmt = 'yuv420p'

for frame in frames:
    av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
    for packet in stream.encode(av_frame):
        container.mux(packet)

# Flush
for packet in stream.encode():
    container.mux(packet)
container.close()
```

---

## 🎯 IMPLEMENTATION GUIDE FOR LLM

### Step 1: Replace VideoWriter Initialization
**File:** `video_processor.py:144-147`

**Remove:**
```python
fourcc = cv2.VideoWriter_fourcc(*'avc1')
right_writer = cv2.VideoWriter(str(anatomical_right_output), fourcc, fps, (width, height))
left_writer = cv2.VideoWriter(str(anatomical_left_output), fourcc, fps, (width, height))
debug_writer = cv2.VideoWriter(str(debug_output), fourcc, fps, (width, height))
```

**Replace with:**
```python
# Start FFmpeg processes for each output video
right_writer = FFmpegWriter(str(anatomical_right_output), width, height, fps)
left_writer = FFmpegWriter(str(anatomical_left_output), width, height, fps)
debug_writer = FFmpegWriter(str(debug_output), width, height, fps)
```

### Step 2: Create FFmpegWriter Class
**Add to:** `video_processor.py` (after imports)

```python
class FFmpegWriter:
    """Fast video writer using FFmpeg with hardware acceleration"""

    def __init__(self, output_path, width, height, fps):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps

        # Detect best encoder (hardware > software)
        encoder = self._detect_best_encoder()

        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-an',
            '-vcodec', encoder,
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            str(output_path)
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # Large buffer
        )

    def _detect_best_encoder(self):
        """Auto-detect best available encoder"""
        import platform

        if platform.system() == 'Darwin':  # macOS
            # Test if VideoToolbox is available
            test_cmd = ['ffmpeg', '-encoders']
            try:
                output = subprocess.check_output(test_cmd, stderr=subprocess.DEVNULL, text=True)
                if 'h264_videotoolbox' in output:
                    return 'h264_videotoolbox'
            except:
                pass

        # Fallback to fast software encoder
        return 'libx264'

    def write(self, frame):
        """Write a frame (fast - just pipes to ffmpeg)"""
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # FFmpeg process died, check for errors
            stderr = self.process.stderr.read()
            raise RuntimeError(f"FFmpeg died: {stderr}")

    def release(self):
        """Finish writing and close"""
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()

        # Check for errors
        if self.process.returncode != 0:
            stderr = self.process.stderr.read()
            print(f"FFmpeg warning for {self.output_path}: {stderr}")
```

### Step 3: Update Frame Writing
**File:** `video_processor.py:280-281`

**No changes needed!** The `write()` method has the same interface:
```python
right_writer.write(right_face.astype(np.uint8))  # Same as before
```

### Step 4: Update Cleanup
**File:** `video_processor.py:385-388`

**No changes needed!** The `release()` method has the same interface:
```python
right_writer.release()  # Same as before
```

---

## 🔧 TESTING & VALIDATION

### Expected Performance Improvements:
- **Write speed**: 46ms/frame → <1ms/frame (98% faster)
- **Batch transitions**: Instant (no blocking)
- **Total processing time**: 78s → ~60s (23% faster)
- **Memory usage**: Lower (smaller queue needed)

### Validation Tests:
1. **Run on test video** - Ensure output videos play correctly
2. **Check framerates** - Should see >20 fps during processing
3. **Verify quality** - CRF 18 should match or exceed VideoWriter quality
4. **Test on different systems** - Hardware encoder may vary

### Fallback Strategy:
If FFmpeg approach fails:
1. **Detect encoder availability** in `_detect_best_encoder()`
2. **Fall back to cv2.VideoWriter** with warning message
3. **Log which encoder is being used** for debugging

---

## 📚 ADDITIONAL RESOURCES

### Hardware Encoder Documentation:
- **VideoToolbox (macOS)**: https://trac.ffmpeg.org/wiki/HWAccelIntro#VideoToolbox
- **NVENC (NVIDIA)**: https://trac.ffmpeg.org/wiki/HWAccelIntro#NVENC
- **QSV (Intel)**: https://trac.ffmpeg.org/wiki/Hardware/QuickSync

### FFmpeg Quality Settings:
- **CRF**: 0-51, lower = better quality (18 = visually lossless)
- **Preset**: ultrafast, superfast, veryfast, faster, fast, medium, slow
- **Pixel format**: yuv420p (most compatible)

### Alternative Libraries:
- **imageio**: `pip install imageio imageio-ffmpeg`
- **PyAV**: `pip install av`
- **ffmpeg-python**: `pip install ffmpeg-python`

---

## 🎯 PRIORITY & IMPACT

**Priority:** HIGH
**Impact:** 98% reduction in write overhead, eliminates batch transition delays
**Complexity:** MEDIUM (requires FFmpeg subprocess management)
**Risk:** LOW (can fall back to cv2.VideoWriter if FFmpeg fails)

---

## ✅ SUCCESS CRITERIA

The implementation is successful when:
1. ✅ Batch processing shows **no write delays** (<0.01s per batch)
2. ✅ Frame processing speed is **>20 fps**
3. ✅ Output videos are **playable and correct quality**
4. ✅ Total processing time is **<65s** for 1000-frame video
5. ✅ Console shows **"VideoToolbox"** or **"hardware"** encoder being used

---

## 🚀 NEXT STEPS FOR IMPLEMENTING LLM

1. Read this document fully
2. Implement `FFmpegWriter` class as specified above
3. Replace cv2.VideoWriter initialization in `video_processor.py:144-147`
4. Test on sample video
5. Verify performance metrics match expectations
6. If successful, apply same pattern to OpenFace AU extraction video writing (if applicable)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-16
**Author:** Claude (Anthropic)
**Context:** S1 Face Mirror Pipeline Optimization Project
