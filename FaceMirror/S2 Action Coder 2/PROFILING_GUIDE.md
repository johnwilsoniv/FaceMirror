# Performance Profiling Guide

This guide explains how to use the two complementary profiling approaches to diagnose performance issues in Action Coder.

## Overview

We have implemented two profiling approaches that work together to give you a complete picture:

1. **Diagnostic Profiler (Option A)** - Component-level timing and cache analysis
2. **cProfile (Option C)** - Function-level call graph and bottleneck identification

Use **both** approaches together for maximum insight!

---

## Option A: Diagnostic Profiler (Component-Level Analysis)

### What It Does
- Tracks component-level timing (frame seek, frame read, color conversion, etc.)
- Monitors cache hit/miss rates for both RGB and QImage caches
- Records paint event breakdown
- Identifies top bottlenecks by total time spent

### How to Use

**Step 1:** Run the application normally
```bash
cd "S2 Action Coder"
python3 main.py
```

**Step 2:** Use the application for 1-2 minutes:
- Load a video file
- Play for 10-15 seconds
- Scrub the timeline (click to seek multiple times)
- Pause/resume playback a few times

**Step 3:** Close the application

**Step 4:** Check the generated reports
- **Console output:** You'll see a summary printed to the terminal
- **JSON report:** `diagnostic_report_YYYYMMDD_HHMMSS.json`

### What to Look For

#### Cache Performance
```
📊 CACHE PERFORMANCE:
  RGB Cache:    85.5% hit rate (342 hits / 400 total)
  QImage Cache: 92.1% hit rate (368 hits / 400 total)
  Time Saved:   5234.2ms total
```
- **Good:** RGB cache >80%, QImage cache >90%
- **Bad:** Hit rates <70% indicate cache is too small or thrashing

#### Top Bottlenecks
```
🔥 TOP BOTTLENECKS (by total time):
  1. video.total_frame_extraction
     Total: 12543.2ms | Avg: 31.36ms | Max: 156.23ms | Count: 400
  2. video.frame_read
     Total: 8234.1ms | Avg: 20.59ms | Max: 98.45ms | Count: 400
```
- **Total time** shows where the app spends the MOST cumulative time
- **Average time** shows typical operation duration
- **Max time** shows worst-case spikes

#### Component Breakdown
```
🎬 FRAME EXTRACTION BREAKDOWN:
  frame_seek: avg=8.23ms, max=45.12ms
  frame_read: avg=20.59ms, max=98.45ms
  color_conversion: avg=2.54ms, max=12.33ms
  qimage_conversion: avg=3.12ms, max=15.67ms
```
- Shows exactly where time is spent in video frame processing
- Helps identify which specific operation is slow

---

## Option C: cProfile (Function-Level Analysis)

### What It Does
- Profiles ALL function calls in the application
- Identifies hot spots (most frequently called functions)
- Shows call graphs (which functions call which)
- Provides cumulative vs internal time breakdown

### How to Use

**Step 1:** Run the application with cProfile wrapper
```bash
cd "S2 Action Coder"
python3 run_cprofile_analysis.py
```

**Step 2:** Use the application for 1-2 minutes:
- Load a video file
- Play for 10-15 seconds
- Scrub the timeline (click to seek multiple times)
- Pause/resume playback a few times

**Step 3:** Close the application

**Step 4:** Check the generated reports
- `cprofile_cumulative_YYYYMMDD_HHMMSS.txt` - Top bottlenecks by cumulative time
- `cprofile_tottime_YYYYMMDD_HHMMSS.txt` - Top functions by internal time
- `cprofile_video_player_YYYYMMDD_HHMMSS.txt` - Video player specific analysis
- `cprofile_timeline_YYYYMMDD_HHMMSS.txt` - Timeline widget specific analysis
- `cprofile_call_counts_YYYYMMDD_HHMMSS.txt` - Most frequently called functions

### What to Look For

#### Cumulative Time Report (Overall Bottlenecks)
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   400    0.234    0.001   12.543    0.031 qt_media_player.py:251(get_frame)
   400    8.234    0.021    8.234    0.021 {method 'read' of 'cv2.VideoCapture'}
```
- **cumtime** (cumulative time) = time in function + all functions it calls
- Look for functions with high cumtime - these are the bottlenecks

#### Internal Time Report (Pure Function Time)
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   400    8.234    0.021    8.234    0.021 {method 'read' of 'cv2.VideoCapture'}
   400    2.543    0.006    2.543    0.006 {method 'cvtColor' in 'cv2'}
```
- **tottime** (total time) = time spent in function itself only
- Shows which individual operations are inherently slow

#### Call Counts (Optimization Opportunities)
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 50000    0.123    0.000    0.456    0.000 timeline_widget.py:456(update)
```
- Functions called 10,000+ times are candidates for optimization
- Even small improvements (0.1ms → 0.05ms) have huge impact

---

## Using Both Together

### Recommended Workflow

1. **Run Diagnostic Profiler first** (main.py with profiling enabled)
   - Identifies which **components** are slow (e.g., "frame_read" vs "color_conversion")
   - Shows cache performance

2. **Run cProfile second** (run_cprofile_analysis.py)
   - Identifies which **specific functions** within those components are the problem
   - Shows call graphs to understand why functions are being called

3. **Cross-reference the results**
   - Diagnostic Profiler: "video.frame_read takes avg 20ms"
   - cProfile: "cv2.VideoCapture.read() is called 400 times, cumtime=8234ms"
   - **Conclusion:** Frame reading is the bottleneck (not seek, not conversion)

### Example Analysis

**Scenario:** Application feels laggy during playback

**Step 1: Check Diagnostic Profiler**
```
🔥 TOP BOTTLENECKS:
  1. video.total_frame_extraction: Total: 12543ms | Avg: 31ms
  2. video.frame_read: Total: 8234ms | Avg: 20ms
  3. video.qimage_conversion: Total: 1248ms | Avg: 3ms
```
→ Frame extraction is the problem, specifically frame reading

**Step 2: Check cProfile Cumulative Time**
```
   400    0.234    0.001   12.543    0.031 qt_media_player.py:251(get_frame)
   400    8.234    0.021    8.234    0.021 {method 'read' of 'cv2.VideoCapture'}
```
→ cv2.VideoCapture.read() is taking 20ms per call

**Step 3: Check Cache Stats**
```
📊 CACHE PERFORMANCE:
  RGB Cache: 45.2% hit rate (181 hits / 400 total)
```
→ Cache hit rate is too low! Only hitting cache 45% of the time

**Conclusion:**
- Frame reading (cv2.VideoCapture.read) is the bottleneck at 20ms/frame
- Cache is undersized - too many cache misses forcing repeated disk reads
- **Solution:** Increase cache size from 50 → 100 frames

---

## Performance Targets

### Acceptable Performance
- **Frame extraction:** <16ms average (for 60 FPS UI)
- **RGB cache hit rate:** >80%
- **QImage cache hit rate:** >90%
- **Event loop latency:** <16ms average
- **Paint events:** <16ms (timeline, video widget)

### Problem Indicators
- **Frame extraction >30ms:** Video decoding bottleneck
- **Cache hit rate <70%:** Cache too small or thrashing
- **Paint events >50ms:** UI rendering bottleneck
- **Event loop latency >30ms:** General responsiveness issues

---

## Common Bottlenecks & Solutions

### 1. Slow Frame Extraction (30-100ms)
**Symptoms:**
- Diagnostic Profiler shows `video.frame_read` >30ms
- cProfile shows `cv2.VideoCapture.read()` has high cumtime

**Solutions:**
- Increase cache size (less disk I/O)
- Use lower resolution video for testing
- Consider video transcoding to more efficient codec

### 2. Low Cache Hit Rate (<70%)
**Symptoms:**
- Diagnostic Profiler shows RGB/QImage cache hit rate <70%
- Frequent cache misses forcing re-extraction

**Solutions:**
- Increase `max_cache_size` in qt_media_player.py:116
- Increase `max_qimage_cache_size` in qt_media_player.py:117
- Monitor memory usage to avoid OOM

### 3. Slow Paint Events (>50ms)
**Symptoms:**
- Performance profiler shows timeline paintEvent >50ms
- UI feels choppy during playback

**Solutions:**
- Reduce number of items drawn per paint
- Use QPainter caching for static elements
- Implement viewport culling (only draw visible items)

### 4. High Event Loop Latency (>30ms)
**Symptoms:**
- Performance profiler shows avg latency >30ms
- UI freezes during operations

**Solutions:**
- Move heavy operations to background threads
- Break up long-running tasks into smaller chunks
- Reduce signal/slot overhead

---

## Files Reference

### Created Files
- `diagnostic_profiler.py` - Component-level profiling system
- `run_cprofile_analysis.py` - cProfile wrapper script
- `PROFILING_GUIDE.md` - This guide

### Modified Files
- `qt_media_player.py` - Added diagnostic profiler instrumentation
- `main.py` - Added diagnostic profiler report generation on exit

### Generated Reports
- `diagnostic_report_YYYYMMDD_HHMMSS.json` - Diagnostic profiler output
- `cprofile_cumulative_YYYYMMDD_HHMMSS.txt` - Cumulative time analysis
- `cprofile_tottime_YYYYMMDD_HHMMSS.txt` - Internal time analysis
- `cprofile_video_player_YYYYMMDD_HHMMSS.txt` - Video player analysis
- `cprofile_timeline_YYYYMMDD_HHMMSS.txt` - Timeline analysis
- `cprofile_call_counts_YYYYMMDD_HHMMSS.txt` - Call frequency analysis

---

## Next Steps

After gathering profiling data:

1. **Identify the top bottleneck** from diagnostic profiler
2. **Confirm with cProfile** which specific function is responsible
3. **Check if it's fixable** (our code vs library limitation)
4. **Implement targeted optimization** based on data
5. **Re-profile to measure improvement**

Remember: **Measure, don't guess!** Always use profiling data to guide optimizations.
