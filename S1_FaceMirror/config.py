"""
S1 Face Mirror - Production Configuration

This module contains all configurable settings for the Face Mirror application.
Modify these values to adjust performance, behavior, and output settings.
"""

import os

# ============================================================================
# Application Metadata
# ============================================================================

VERSION = "1.0.0"
APP_NAME = "S1 Face Mirror"

# ============================================================================
# Performance Settings
# ============================================================================

# Number of worker threads for parallel frame processing
# Recommended: 4-6 for 8-10 core systems, 2-4 for 4-6 core systems
NUM_THREADS = 6

# Frames loaded per batch (controls memory usage)
# Lower values use less memory but may reduce throughput
# Recommended: 100 for 16GB+ RAM, 50 for 8GB RAM
BATCH_SIZE = 100

# Deep memory cleanup interval (every N videos)
# Lower values = more frequent cleanup, higher values = better performance
MEMORY_CHECKPOINT_INTERVAL = 10

# Terminal progress update interval (every N frames)
# Higher values = less console spam
PROGRESS_UPDATE_INTERVAL = 50

# ============================================================================
# Model & Detection Settings
# ============================================================================

# Enable AU45 (blink) detection
# Warning: Enabling AU45 requires landmark detection on every frame
# This reduces performance from ~14-28 FPS to ~2-3 FPS
# Set to False if you don't need blink detection (5-7x speedup)
ENABLE_AU45_CALCULATION = True

# Face detection confidence threshold (0.0 - 1.0)
# Lower values detect more faces but increase false positives
CONFIDENCE_THRESHOLD = 0.5

# Non-maximum suppression threshold for face detection (0.0 - 1.0)
# Lower values = more aggressive duplicate suppression
NMS_THRESHOLD = 0.4

# Visibility threshold for face filtering (0.0 - 1.0)
# Faces with confidence below this are discarded
VIS_THRESHOLD = 0.5

# ============================================================================
# System Threading Configuration
# ============================================================================
# These environment variables control low-level library threading
# They are set before imports to prevent thread contention
# Total threads ≈ NUM_THREADS × (OMP + ONNX) ≈ 6 × 4 = 24 threads

OMP_NUM_THREADS = 2          # OpenMP threads (PyTorch)
MKL_NUM_THREADS = 2          # Intel MKL threads
OPENBLAS_NUM_THREADS = 2     # OpenBLAS threads
VECLIB_MAXIMUM_THREADS = 2   # macOS Accelerate framework
NUMEXPR_NUM_THREADS = 2      # NumExpr threads

# ============================================================================
# Performance Profiling
# ============================================================================

# Enable detailed performance profiling
# When enabled, creates timestamped reports with:
#   - Per-operation timing breakdown
#   - FPS statistics
#   - Bottleneck identification
# Warning: Adds ~1-2% overhead
ENABLE_PROFILING = False

# Directory for profiling reports (None = Desktop)
# Uses same output directory as processed videos
PROFILING_OUTPUT_DIR = "/Users/johnwilsoniv/Documents/SplitFace/logs"

# ============================================================================
# Logging Configuration
# ============================================================================

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
# DEBUG: Verbose diagnostic output (development)
# INFO: Normal operation messages (recommended)
# WARNING: Important warnings only
# ERROR: Errors only
LOG_LEVEL = "INFO"

# Optional log file path (None = console only)
# Set to a file path to enable file logging
# Example: LOG_FILE = os.path.expanduser("~/Desktop/face_mirror.log")
LOG_FILE = None

# ============================================================================
# GPU Acceleration Settings (pyclnf / pymtcnn)
# ============================================================================

# Enable GPU acceleration for CLNF landmark detection
# When enabled, uses MPS (Apple Silicon) or CUDA (NVIDIA) for response maps
# Provides ~3x speedup over CPU-only processing
USE_GPU = True

# GPU device selection
# Options: 'auto', 'mps', 'cuda', 'cpu'
# 'auto' selects best available (MPS on Apple Silicon, CUDA on NVIDIA, else CPU)
GPU_DEVICE = 'auto'

# CLNF convergence profile for video processing
# Options: 'video' (faster, relies on temporal continuity), 'image' (more robust)
CLNF_CONVERGENCE_PROFILE = 'video'

# Enable eye landmark refinement
# Improves accuracy for AU45 (blink) detection
USE_EYE_REFINEMENT = True

# Face detection interval (frames)
# Re-detect faces every N frames (0 = detect only on first frame)
# Lower values = more robust to movement, higher values = faster
FACE_DETECTION_INTERVAL = 60

# Generate debug video output (set to False for production to save ~0.3-0.5 fps)
GENERATE_DEBUG_VIDEO = True

# PyMTCNN backend for face detection
# Options: 'auto', 'coreml', 'cuda', 'cpu'
# 'coreml' is fastest on Apple Silicon
MTCNN_BACKEND = 'coreml'

# ============================================================================
# Device Selection Strategy (Legacy)
# ============================================================================

# Device preference order (do not modify unless you know what you're doing)
# 1. CUDA (NVIDIA GPU) - Best for NVIDIA hardware
# 2. ONNX+CoreML (Apple Silicon) - Best for M1/M2/M3 Macs
# 3. ONNX+CPU (Intel CPU) - Best for Intel Macs without GPU
# 4. PyTorch CPU - Fallback for all platforms

# Force specific device (None = auto-detect)
# Options: None, 'cuda', 'cpu'
# Warning: Forcing a device may reduce performance
FORCE_DEVICE = None

# ============================================================================
# Garbage Collection Optimization
# ============================================================================

# Garbage collection threshold (generation 0)
# Higher values reduce GC overhead but increase memory usage
# Default Python: 700, Optimized: 10000
# Research shows this reduces GC from ~3% to ~0.5% of runtime
GC_THRESHOLD_GEN0 = 10000
GC_THRESHOLD_GEN1 = 10
GC_THRESHOLD_GEN2 = 10


def apply_environment_settings():
    """
    Apply environment variable settings before library imports.
    This function should be called at the very start of the application.
    """
    os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(MKL_NUM_THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(OPENBLAS_NUM_THREADS)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(VECLIB_MAXIMUM_THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUMEXPR_NUM_THREADS)


def get_profiling_output_dir():
    """Get the directory for profiling output files."""
    if PROFILING_OUTPUT_DIR is None:
        return os.path.expanduser("~/Desktop")
    return PROFILING_OUTPUT_DIR
