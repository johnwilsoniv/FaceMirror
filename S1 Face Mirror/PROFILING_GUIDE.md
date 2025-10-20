# Performance Profiling Guide - Face Mirror v2.0.0

This guide explains how to profile Face Mirror performance to identify bottlenecks and optimize performance.

## Table of Contents
1. [Built-in Custom Profiler (Recommended)](#built-in-custom-profiler-recommended)
2. [Xcode Instruments (System-level profiling)](#xcode-instruments-system-level-profiling)
3. [Interpreting Profiling Results](#interpreting-profiling-results)
4. [Common Performance Bottlenecks](#common-performance-bottlenecks)

---

## Built-in Custom Profiler (Recommended)

### What It Does

The custom profiler provides **detailed, code-level timing** specifically instrumented for Face Mirror:

✓ **Model inference timing** - Exact time spent in RetinaFace, STAR, and MTL models
✓ **Neural Engine vs CPU breakdown** - Shows which operations run on CoreML/Neural Engine
✓ **Preprocessing/postprocessing overhead** - Identifies non-model bottlenecks
✓ **Memory operation tracking** - Frame reading, writing, and copying times
✓ **Per-operation statistics** - Min, max, average, and total time for each operation

### How to Use

**The profiler is already integrated and runs automatically!**

Simply run Face Mirror normally:

```bash
python main.py
```

At the end of processing, you'll see a detailed report:

```
================================================================================
DETAILED PERFORMANCE PROFILING
================================================================================

Total Profiled Time: 235.478s

────────────────────────────────────────────────────────────────────────────────
Category: MODEL_INFERENCE
────────────────────────────────────────────────────────────────────────────────
Total Time: 198.234s (84.2% of total)

Operation                      Count    Total        Avg        Min        Max      %
------------------------------ -------- ---------- ---------- ---------- ---------- ------
MTL_coreml                      1942   112.345s    57.8ms     52.1ms     89.3ms   56.7%
RetinaFace_coreml                190    48.672s   256.2ms    248.6ms    312.1ms   24.5%
STAR_coreml                       48    37.217s   775.4ms    720.3ms    890.2ms   18.8%

────────────────────────────────────────────────────────────────────────────────
Category: PREPROCESSING
────────────────────────────────────────────────────────────────────────────────
Total Time: 23.156s (9.8% of total)
...
```

### What to Look For

1. **Model Inference Category** - Shows time spent in Neural Engine/CoreML
   - `MTL_coreml` - AU extraction (most common operation)
   - `RetinaFace_coreml` - Face detection
   - `STAR_coreml` - Landmark detection (if enabled)

2. **Preprocessing/Postprocessing** - CPU-bound image operations
   - High values here indicate bottlenecks in image processing

3. **Percentage Breakdown** - Identifies which operation dominates
   - Example: "MTL_coreml: 56.7%" means over half the time is AU extraction

### Advanced: Export to JSON

Uncomment line 655 in `main.py` to export profiling data:

```python
profiler.export_json("performance_profile.json")
```

This creates a JSON file with all timing data for further analysis.

---

## Xcode Instruments (System-level profiling)

### What It Does

Xcode Instruments provides **system-level profiling** with macOS integration:

✓ CPU usage per thread
✓ Memory allocations and leaks
✓ GPU/Neural Engine utilization (limited visibility)
✓ System call tracing
✓ Energy usage

### Pros vs Cons vs Custom Profiler

| Feature | Custom Profiler | Xcode Instruments |
|---------|----------------|-------------------|
| **Model inference timing** | ✓ Exact timing | ✗ Black box |
| **Neural Engine breakdown** | ✓ Shows CoreML ops | ✗ Hidden from userspace |
| **Easy setup** | ✓ Automatic | ⚠ Requires Xcode |
| **Code-specific data** | ✓ Face Mirror operations | ✗ Generic system view |
| **Memory tracking** | ✓ Per operation | ✓ System-wide |
| **CPU thread usage** | ✗ | ✓ Detailed |
| **Learning curve** | ✓ Simple reports | ⚠ Complex interface |

### How to Use Xcode Instruments

1. **Install Xcode** (if not already installed):
   ```bash
   xcode-select --install
   ```

2. **Launch Instruments**:
   ```bash
   open -a Instruments
   ```

3. **Select a template**:
   - **Time Profiler** - CPU usage breakdown (most useful)
   - **Allocations** - Memory usage and leaks
   - **System Trace** - Comprehensive system analysis

4. **Profile your Python script**:
   - Click "Choose Target" → "Choose Target..."
   - Navigate to: `/usr/local/bin/python3.10` (or your Python path)
   - Set arguments: `/path/to/main.py`
   - Click Record (red button)

5. **Analyze results**:
   - Instruments will show CPU usage, function calls, and threads
   - Look for "hot spots" (functions using most CPU time)
   - Neural Engine operations appear as `ANE_*` framework calls (limited visibility)

### Limitations

⚠ **CoreML/Neural Engine operations are opaque** in Instruments. You'll see:
- Time spent in `CoreML.framework`
- But NOT which specific operations are slow
- This is why the custom profiler is more useful for Face Mirror

---

## Interpreting Profiling Results

### Example Analysis

```
MTL_coreml:       112.3s (56.7% of total)  <-- BOTTLENECK
RetinaFace_coreml: 48.7s (24.5%)
STAR_coreml:       37.2s (18.8%)
```

**Interpretation**:
- MTL (AU extraction) dominates (56.7%)
- Each MTL call takes ~58ms (from Avg column)
- 1942 calls total (from Count column)

**Optimization Priority**:
1. **HIGH**: Optimize MTL model (biggest impact)
2. **MEDIUM**: Optimize RetinaFace (24.5% of time)
3. **LOW**: STAR is fine (only 18.8%)

### Comparing CoreML vs CPU

The profiler shows operation names like:
- `MTL_coreml` - Running on Neural Engine (fast, ~58ms)
- `MTL_onnx_cpu` - Running on CPU (slower, ~180ms)

If you see `onnx_cpu` instead of `coreml`, it means CoreML is not being used!

**To fix**: Check CoreML warnings in console:
```
2025-10-17 09:53:40.090855 [W:onnxruntime:, coreml_execution_provider.cc:107]
CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 28
```

**High partition count (28) = fragmented execution = slower performance**

---

## Common Performance Bottlenecks

### 1. MTL Model Dominates (56%+ of time)

**Why**: MTL has only 69% Neural Engine coverage (31% CPU fallback)

**Fix Options**:
- Re-export MTL model with better CoreML compatibility
- Use newer ONNX→CoreML converter
- Simplify model architecture (trade accuracy for speed)

**Expected Gain**: 15-25% overall speedup

---

### 2. High Preprocessing Time (>10% of total)

**Why**: CPU-bound image resizing/normalization

**Fix Options**:
- Use GPU-accelerated CV operations
- Reduce preprocessing complexity
- Batch preprocessing operations

**Expected Gain**: 5-10% overall speedup

---

### 3. Face Detection Too Slow (>30% of time)

**Why**: Either running too frequently or using CPU fallback

**Fix Options**:
- Increase detection interval (currently every ~5 frames)
- Check CoreML is enabled (should see `RetinaFace_coreml`, not `onnx_cpu`)
- Use faster face detector (MobileNet is already lightweight)

**Expected Gain**: 10-20% overall speedup

---

### 4. Memory Operations High (>5% of total)

**Why**: Video I/O bottleneck (reading/writing frames)

**Fix Options**:
- Already using FFmpeg hardware encoding (optimal)
- Increase batch size (trades memory for speed)
- Use SSD instead of HDD for video files

**Expected Gain**: Minimal (already optimized)

---

## Profiling Best Practices

### 1. Run on Representative Videos
- Use typical video length (10-30 seconds)
- Use actual resolution (1080p or 4K)
- Avoid tiny test videos (profiling overhead dominates)

### 2. Multiple Runs for Accuracy
- First run: ~10% slower (model warm-up)
- Subsequent runs: Consistent timing
- Average 3 runs for reliable data

### 3. Profile Production Settings
- Don't profile debug builds
- Use actual hardware (not VM)
- Close other applications (CPU/GPU contention)

### 4. Focus on Biggest Bottlenecks First
- Optimize operations >20% of total time first
- Ignore operations <5% of total time
- Use 80/20 rule: 20% of code = 80% of time

---

## Quick Reference: Profiling Commands

```bash
# Run with built-in profiler (automatic)
python main.py

# Export profiling data to JSON (uncomment line 655 in main.py)
# Result: performance_profile.json in working directory

# Profile with Xcode Instruments (requires Xcode)
instruments -t "Time Profiler" -D profile_output.trace \
    /usr/local/bin/python3.10 /path/to/main.py

# View Instruments trace file
open profile_output.trace
```

---

## Appendix: Performance Profile from Sample Run

Based on your log output:

```
AU Extraction: ~113ms per frame (8.6-8.8 FPS)
  └─ MTL model: ~95% of time (~107ms per frame)
     ├─ Neural Engine: ~69% coverage (~74ms)
     └─ CPU fallback: ~31% (~33ms)
     └─ Partitions: 28 (high fragmentation)

Face Detection: ~248.6ms per detection (every ~5 frames)
  └─ RetinaFace: 81.3% Neural Engine coverage
     └─ Partitions: 3 (low fragmentation, good)

Landmark Detection: ~183ms per frame (when enabled)
  └─ STAR: 82.6% Neural Engine coverage
     └─ Partitions: 18 (moderate fragmentation)
```

**Priority 1**: Reduce MTL partitions (28 → <10) for 20-30% speedup
**Priority 2**: Keep face detection optimized (already good)
**Priority 3**: STAR is acceptable if only used occasionally

---

## Support

For questions about profiling or performance optimization:
1. Check `COREML_PERFORMANCE_ANALYSIS.md` for CoreML-specific insights
2. Review `TECHNICAL_OVERVIEW.md` for architecture details
3. Profile with built-in profiler first (faster iteration)
4. Use Xcode Instruments only for system-level issues (memory leaks, threading)

**Remember**: The custom profiler gives you exactly what you need for Face Mirror optimization!
