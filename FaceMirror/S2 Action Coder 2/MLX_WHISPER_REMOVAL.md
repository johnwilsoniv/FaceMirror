# MLX-Whisper Support Removal

## Summary

MLX-Whisper support has been completely removed from S2 Action Coder in favor of faster-whisper for better transcription accuracy.

## Reason for Removal

MLX-Whisper, while significantly faster on Apple Silicon (5-10x speedup), had critical accuracy limitations:

1. **No beam search support** - Used beam_size=1 (default) instead of beam_size=5
2. **No VAD (Voice Activity Detection)** - Could not filter out silence and background noise
3. **No VAD parameters** - Could not tune silence detection thresholds

These limitations caused:
- Lower quality transcriptions
- Hallucinations (model inventing text from background noise)
- Inconsistent accuracy across videos

## What Changed

### whisper_handler.py

**Removed:**
- All MLX-Whisper import attempts
- `MLX_WHISPER_AVAILABLE` flag and checks
- `IS_APPLE_SILICON` detection
- `self.use_mlx` instance variable
- Entire MLX-Whisper transcription branch (60+ lines)
- Conditional logic for choosing between MLX and faster-whisper

**Kept:**
- faster-whisper as the only transcription engine
- Full VAD filtering with configurable parameters
- beam_size=5 for better quality
- CPU parallelization optimizations
- Batch processing support
- All accuracy-focused parameters

### main.py

**No changes needed** - Was already using faster-whisper for model preloading

### Documentation

**Removed:**
- `WHISPER_ACCURACY_ISSUE.md` (no longer relevant)

**Updated:**
- `WHISPER_OPTIMIZATIONS.md` (references faster-whisper only)

## New Behavior

### On Startup

**Old message:**
```
Whisper Handler: MLX-Whisper found (Apple Silicon optimized)
  Expected: 5-10x speedup vs CPU-based inference
  ⚠️  WARNING: MLX-Whisper has lower accuracy than faster-whisper
```

**New message:**
```
Whisper Handler: faster-whisper library loaded successfully.
  Using faster-whisper for maximum accuracy with VAD filtering & beam search
```

### Transcription Quality

All videos now use:
- ✅ beam_size=5 (better decoding quality)
- ✅ VAD filtering enabled (filters silence/noise)
- ✅ Configurable VAD parameters
- ✅ Consistent accuracy across all videos

### Performance

**Apple Silicon (M-series) CPUs:**
- Slower than MLX-Whisper (no longer 5-10x speedup)
- But still optimized with CPU parallelization (1.5-2x vs baseline)
- Accuracy is significantly better

**Intel CPUs:**
- Same performance as before (already used faster-whisper)

**CUDA GPUs:**
- Same performance as before (already used faster-whisper)
- Flash Attention support available

## Migration

**For users:**
- No action required!
- If mlx-whisper is installed, it will simply be ignored
- Optionally uninstall: `pip uninstall mlx-whisper`

**For developers:**
- Code is now simpler (single transcription path)
- No conditional logic for model selection
- Easier to maintain and debug

## Code Changes

### Files Modified

1. **whisper_handler.py**
   - Lines 13-33: Removed MLX-Whisper import and detection
   - Lines 121-142: Removed MLX-specific initialization
   - Lines 198-261: Removed entire MLX-Whisper transcription branch
   - Fixed indentation throughout

2. **Documentation**
   - Removed `WHISPER_ACCURACY_ISSUE.md`

### Lines of Code Removed

- ~80 lines of MLX-specific code
- ~30 lines of conditional logic
- Total: ~110 lines removed

## Testing

### Verification Steps

1. Start S2 Action Coder
2. Load a video with background noise
3. Check console output:
   - Should say "Using faster-whisper for maximum accuracy"
   - Should show "Using VAD Filter: True"
4. Check transcription quality:
   - No hallucinations from background noise
   - Consistent quality across videos
   - Better detection of quiet speech

### Expected Results

- ✅ All videos use faster-whisper
- ✅ VAD filtering active
- ✅ Better accuracy overall
- ⚠️ Slower processing on Apple Silicon (acceptable tradeoff)

## Benefits

### For Accuracy

- ✅ Consistent quality across all videos
- ✅ No hallucinations from noise
- ✅ Better handling of silence
- ✅ Configurable VAD thresholds
- ✅ Beam search for better decoding

### For Maintainability

- ✅ Simpler code (single transcription path)
- ✅ Fewer conditionals and branches
- ✅ Easier to debug
- ✅ Less complexity
- ✅ One library to maintain

### For Users

- ✅ Better transcription quality
- ✅ More predictable results
- ✅ No need to choose speed vs accuracy
- ✅ Consistent behavior

## Considerations

### Speed Tradeoff

**Apple Silicon users will notice:**
- Slower transcription (no longer 5-10x speedup from MLX)
- But still optimized with CPU parallelization
- Processing time is acceptable for most use cases

**If speed is critical:**
- Consider using a CUDA GPU machine instead
- Or accept the CPU processing time for better accuracy
- Remember: accuracy is more important than speed for research

### Future Options

If MLX-Whisper adds support for:
- beam_size parameter
- VAD filtering
- vad_parameters configuration

We could consider re-adding it as an optional faster mode.

## Summary

**Before:** Two transcription engines, lower accuracy on Apple Silicon
**After:** One transcription engine, consistent high accuracy everywhere

The removal of MLX-Whisper simplifies the codebase and ensures all users get the same high-quality transcription results, regardless of their hardware.
