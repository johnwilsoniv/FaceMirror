"""
Compare Python and C++ response maps for landmark 36
"""
import numpy as np

# Load Python response map
python_resp = np.load('/tmp/python_response_map_lm36_iter0_ws11.npy')

# Load C++ response map
with open('/tmp/cpp_response_map_lm36_iter0.bin', 'rb') as f:
    rows = np.fromfile(f, dtype=np.int32, count=1)[0]
    cols = np.fromfile(f, dtype=np.int32, count=1)[0]
    cpp_resp = np.fromfile(f, dtype=np.float32, count=rows*cols).reshape(rows, cols)

print("Python response map:")
print(f"  Shape: {python_resp.shape}")
print(f"  Min: {python_resp.min():.6f}, Max: {python_resp.max():.6f}, Mean: {python_resp.mean():.6f}")
print(f"  Top-left 3x3:\n{python_resp[:3, :3]}")

print("\nC++ response map:")
print(f"  Shape: {cpp_resp.shape}")
print(f"  Min: {cpp_resp.min():.6f}, Max: {cpp_resp.max():.6f}, Mean: {cpp_resp.mean():.6f}")
print(f"  Top-left 3x3:\n{cpp_resp[:3, :3]}")

print("\nDifference:")
diff = cpp_resp - python_resp
print(f"  Absolute difference - Min: {diff.min():.6f}, Max: {diff.max():.6f}, Mean: {diff.mean():.6f}")
print(f"  RMS difference: {np.sqrt(np.mean(diff**2)):.6f}")
print(f"  Correlation: {np.corrcoef(python_resp.flatten(), cpp_resp.flatten())[0,1]:.6f}")

# Find peak locations
py_peak = np.unravel_index(np.argmax(python_resp), python_resp.shape)
cpp_peak = np.unravel_index(np.argmax(cpp_resp), cpp_resp.shape)
print(f"\nPeak locations:")
print(f"  Python: {py_peak}, value: {python_resp[py_peak]:.6f}")
print(f"  C++: {cpp_peak}, value: {cpp_resp[cpp_peak]:.6f}")
