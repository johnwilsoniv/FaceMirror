# CLNF Debug Mode Documentation

## Overview

The CLNF implementation now includes a comprehensive debug mode, similar to MTCNN's `detect_with_debug`. This mode provides detailed insights into the CLNF optimization process, making it easy to:

- Compare Python implementation against C++ OpenFace
- Diagnose convergence issues
- Analyze response maps and mean-shift vectors
- Track landmark positions through iterations
- Understand parameter updates

## Features

### 1. Debug Mode in Optimizer

The `NURLMSOptimizer` class has a new `optimize_with_debug()` method that collects:

- **Initialization**: Initial parameters and landmarks
- **Per-Iteration Data**:
  - Landmark positions
  - Response map peaks and values (for tracked landmarks)
  - Mean-shift vectors
  - Jacobian matrix condition number
  - Parameter updates
  - Shape change metrics
- **Final Results**: Converged parameters and landmarks

### 2. Debug Mode in CLNF Model

The `CLNF` class has a new `fit_with_debug()` method that provides:

- **Initialization**: Bbox, initial parameters, initial landmarks
- **Window Stages**: Debug info for each window size (11, 9, 7)
  - Window size and patch scale
  - Full optimization debug info (from optimizer)
  - Convergence info
- **Final Results**: Total iterations, final landmarks, final parameters

## Usage

### Basic Example

```python
from pyclnf import CLNF
import cv2

# Initialize CLNF (without automatic detector for manual bbox)
clnf = CLNF(detector=None)

# Load image
img = cv2.imread("face.jpg")
bbox = (x, y, w, h)  # Face bounding box

# Run CLNF with debug mode
landmarks, info, debug_info = clnf.fit_with_debug(
    img,
    bbox,
    tracked_landmarks=[36, 48, 30, 8]  # Optional: landmarks to track in detail
)

# Access debug information
print(f"Converged: {info['converged']}")
print(f"Total iterations: {debug_info['final']['total_iterations']}")

# Analyze per-window optimization
for stage in debug_info['window_stages']:
    ws = stage['window_size']
    print(f"\nWindow Size {ws}:")

    # Access per-iteration details
    for iter_data in stage['optimization']['iterations']:
        print(f"  Iteration {iter_data['iteration']}:")

        # Check response maps for tracked landmarks
        for lm_idx, resp_info in iter_data['response_maps'].items():
            print(f"    Landmark {lm_idx}:")
            print(f"      Position: {resp_info['position']}")
            print(f"      Response peak: {resp_info['response_peak']}")
            print(f"      Mean-shift: {resp_info['mean_shift']}")
```

### Saving Debug Output

```python
import json
import numpy as np

# Convert numpy arrays to lists for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

# Save debug info as JSON
debug_json = convert_numpy(debug_info)
with open('clnf_debug.json', 'w') as f:
    json.dump(debug_json, f, indent=2)
```

### Complete Example

See `example_clnf_debug.py` for a complete working example that:
- Loads an image
- Runs CLNF with debug mode
- Prints a summary of debug information
- Saves full debug info as JSON
- Creates a visualization with tracked landmarks

Run it with:
```bash
python example_clnf_debug.py
```

## Debug Info Structure

### Top Level

```python
debug_info = {
    'bbox': (x, y, w, h),                    # Input bounding box
    'tracked_landmarks': [36, 48, 30, 8],    # Landmark indices tracked in detail
    'initialization': {...},                  # Initial state
    'window_stages': [...],                   # Per-window optimization
    'final': {...}                            # Final results
}
```

### Initialization

```python
'initialization': {
    'params': np.ndarray,                    # Initial parameter vector (40,)
    'landmarks': np.ndarray,                 # Initial landmarks (68, 2)
    'tracked_landmarks': {
        36: [x, y],                           # Position of tracked landmarks
        48: [x, y],
        30: [x, y],
        8: [x, y]
    }
}
```

### Window Stages

```python
'window_stages': [
    {
        'window_size': 11,                   # Window size for this stage
        'window_index': 0,                   # Stage index
        'patch_scale': 0.25,                 # Patch expert scale
        'info': {                             # Convergence info
            'converged': True/False,
            'iterations': int,
            'final_update': float
        },
        'optimization': {                     # Detailed optimization debug
            'initialization': {...},          # Same as top-level initialization
            'iterations': [                   # Per-iteration details
                {
                    'iteration': 0,
                    'window_size': 11,
                    'landmarks': np.ndarray,  # (68, 2)
                    'tracked_landmarks': {    # Tracked landmark positions
                        36: [x, y],
                        ...
                    },
                    'response_maps': {        # Response map details for tracked landmarks
                        36: {
                            'position': [x, y],
                            'window_bounds': {
                                'x': [x_min, x_max],
                                'y': [y_min, y_max]
                            },
                            'response_peak': {
                                'row': int,   # Peak location in response map
                                'col': int,
                                'value': float,  # Peak response value
                                'offset_x': float,  # Offset from center
                                'offset_y': float
                            },
                            'response_stats': {
                                'min': float,
                                'max': float,
                                'mean': float
                            },
                            'mean_shift': {
                                'x': float,   # Mean-shift vector
                                'y': float,
                                'magnitude': float
                            }
                        },
                        ... # Other tracked landmarks
                    },
                    'shape_change': float,    # Change from previous iteration
                    'mean_shift_magnitude': float,
                    'jacobian_shape': (136, 40),
                    'jacobian_condition': float,
                    'delta_p': np.ndarray,    # Parameter update
                    'delta_p_magnitude': float,
                    'updated_params': np.ndarray
                },
                ... # More iterations
            ],
            'final': {                        # Final state for this window
                'converged': True/False,
                'iterations_run': int,
                'params': np.ndarray,
                'landmarks': np.ndarray,
                'tracked_landmarks': {...}
            }
        }
    },
    ... # More window stages (11, 9, 7)
]
```

### Final Results

```python
'final': {
    'landmarks': np.ndarray,                 # Final landmarks (68, 2)
    'tracked_landmarks': {                   # Final tracked landmark positions
        36: [x, y],
        48: [x, y],
        30: [x, y],
        8: [x, y]
    },
    'total_iterations': int,                 # Total iterations across all windows
    'params': np.ndarray                     # Final parameter vector
}
```

## Use Cases

### 1. Comparing with C++ OpenFace

Extract response map peaks and mean-shift vectors to compare with C++ debug output:

```python
landmarks, info, debug_info = clnf.fit_with_debug(img, bbox)

# Extract first iteration, first window, landmark 36
stage_0 = debug_info['window_stages'][0]
iter_0 = stage_0['optimization']['iterations'][0]
lm_36 = iter_0['response_maps'][36]

print(f"[PY][ITER0_WS11_LM36] Response peak: ({lm_36['response_peak']['row']}, {lm_36['response_peak']['col']})")
print(f"[PY][ITER0_WS11_LM36] Peak value: {lm_36['response_peak']['value']:.6f}")
print(f"[PY][ITER0_WS11_LM36] Mean-shift: ({lm_36['mean_shift']['x']:.6f}, {lm_36['mean_shift']['y']:.6f})")
```

Compare this output format directly with C++ debug logs.

### 2. Diagnosing Convergence Issues

Track how landmarks move and parameters change:

```python
for stage in debug_info['window_stages']:
    print(f"\nWindow {stage['window_size']}:")
    for iter_data in stage['optimization']['iterations']:
        print(f"  Iter {iter_data['iteration']}: shape_change={iter_data.get('shape_change', 'N/A'):.4f}")
```

### 3. Analyzing Response Maps

Check if response peaks are near the center (good) or at edges (bad):

```python
for lm_idx, resp_info in iter_0['response_maps'].items():
    offset = resp_info['response_peak']
    dist = (offset['offset_x']**2 + offset['offset_y']**2)**0.5
    print(f"Landmark {lm_idx}: peak offset = {dist:.2f} pixels")
    if dist > 3:
        print(f"  ⚠️  Large offset - landmark may not be converging properly")
```

### 4. Tracking Parameter Updates

Monitor how parameters evolve:

```python
for stage in debug_info['window_stages']:
    for iter_data in stage['optimization']['iterations']:
        delta_p_mag = iter_data['delta_p_magnitude']
        print(f"WS{stage['window_size']} Iter{iter_data['iteration']}: Δp={delta_p_mag:.6f}")
```

## Comparison with MTCNN Debug Mode

Similar to MTCNN's `detect_with_debug()`:

```python
# MTCNN
bboxes, landmarks, debug_info = mtcnn.detect(img, return_debug=True)

# CLNF
landmarks, info, debug_info = clnf.fit_with_debug(img, bbox)
```

Both return:
1. Primary results (landmarks)
2. Info dict (convergence, iterations)
3. Debug dict (detailed stage-by-stage information)

## Performance Note

Debug mode has minimal performance impact:
- Only tracked landmarks collect detailed response map info (default: 4 landmarks)
- Debug data collection adds <5% overhead
- Most computation is identical to normal mode

For production use, use the regular `fit()` method.

## See Also

- `example_clnf_debug.py` - Complete working example
- `pyclnf/core/optimizer.py` - Optimizer debug implementation
- `pyclnf/clnf.py` - CLNF debug implementation
- MTCNN debug mode - Similar pattern in `pymtcnn/`
