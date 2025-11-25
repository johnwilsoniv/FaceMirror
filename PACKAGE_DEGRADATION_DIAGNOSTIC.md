# Package Degradation Diagnostic Process

This document describes the systematic process used to identify and fix the pyclnf accuracy degradation. Use this process to check other packages for similar issues.

## Background

When packages are moved from a monorepo to separate repos, critical code can be inadvertently simplified or lost, causing performance degradation.

## Diagnostic Steps

### Step 1: Establish Ground Truth

1. **Find the "working" commit** in git history:
   ```bash
   cd "/path/to/main/repo"
   git log --oneline | grep -i "accuracy\|max\|working"
   ```

2. **Record expected performance** (from commit message or test results):
   - pyclnf expected: ~0.76px mean error, ~3-5px jaw error

3. **Test current performance**:
   ```python
   # Compare Python output to C++ OpenFace ground truth
   errors = np.sqrt(np.sum((cpp_landmarks - py_landmarks)**2, axis=1))
   print(f'Mean error: {np.mean(errors):.2f}px')
   print(f'Jaw (0-16): {np.mean(errors[0:17]):.2f}px')
   ```
   - pyclnf observed: 10-14px mean error, 28px jaw error

### Step 2: Check Git History

1. **List commits affecting the package since "max accuracy"**:
   ```bash
   git log --oneline <max_accuracy_commit>..HEAD -- package_name/
   ```

2. **Check if package was removed from main repo**:
   ```bash
   git diff <max_accuracy_commit> HEAD --stat -- package_name/
   ```
   - If you see all deletions (lines ending with `-`), package was moved

3. **Check the package's own git history** (if it's now a separate repo):
   ```bash
   cd package_name/
   git log --oneline
   ```

### Step 3: Compare Old vs New Code

1. **Extract old files from main repo**:
   ```bash
   git show <max_accuracy_commit>:package/file.py > /tmp/old_file.py
   ```

2. **Compare file sizes**:
   ```bash
   wc -l /tmp/old_file.py current_package/file.py
   ```
   - Significant size difference indicates changes

3. **Diff key files**:
   ```bash
   diff /tmp/old_file.py current_package/file.py | head -200
   ```

### Step 4: Identify Critical Differences

Look for these common degradation patterns:

#### Model/Algorithm Changes
- Different model types (CEN vs CCNF)
- Missing scales/resolutions
- Simplified architectures

**pyclnf example**:
```python
# OLD (working):
from .core.cen_patch_expert import CENModel
self.patch_scaling = [0.25, 0.35, 0.5, 1.0]  # 4 scales

# NEW (degraded):
from .core.patch_expert import CCNFModel
self.patch_scaling = [0.25, 0.35, 0.5]  # 3 scales
```

#### Optimization/Processing Changes
- Missing optimization phases
- Simplified algorithms
- Changed default parameters

**pyclnf example**:
```python
# OLD (working):
# Two-phase optimization: RIGID → NON-RIGID
# Precomputed response maps
regularization=20, max_iterations=5

# NEW (degraded):
# Single optimization loop
# Recomputed response maps each iteration
regularization=25, max_iterations=20
```

#### Default Parameter Changes
- Window sizes, thresholds, tolerances
- Regularization weights
- Iteration counts

### Step 5: Restore Working Code

1. **Extract all critical files from old commit**:
   ```bash
   mkdir -p /tmp/old_package/core
   git show <commit>:package/main.py > /tmp/old_package/main.py
   git show <commit>:package/core/optimizer.py > /tmp/old_package/core/optimizer.py
   # etc.
   ```

2. **Copy to current package structure**:
   ```bash
   cp /tmp/old_package/main.py current_package/package/main.py
   # etc.
   ```

3. **Test imports**:
   ```python
   from package import MainClass
   print('Import successful!')
   ```

4. **Verify accuracy restored**:
   ```python
   # Run same accuracy test as Step 1
   ```

### Step 6: Compare Dependent Components

Check if related packages (e.g., detectors) are still correct:

```bash
# Compare old vs new for dependent package
wc -l /tmp/old_dependent/base.py current_dependent/base.py

# Check for any differences
diff /tmp/old_dependent/base.py current_dependent/base.py | wc -l
```

If output is `0`, files are identical - no restoration needed.

## pyclnf Specific Findings

### Root Causes
1. **CEN → CCNF model switch**: Lost accuracy
2. **Two-phase → Single-phase optimization**: Lost convergence quality
3. **4 scales → 3 scales**: Lost finest resolution
4. **Precomputed → Recomputed response maps**: Different algorithm

### Fix Applied
Restored all core files from commit `3acd1c56`:
- clnf.py
- core/optimizer.py
- core/pdm.py
- core/patch_expert.py
- core/cen_patch_expert.py
- core/utils.py
- utils/retinaface_correction.py

### Results
| Metric | Before | After |
|--------|--------|-------|
| Mean error | 10-14px | 2.54px |
| Jaw (0-16) | 28px | 4.42px |
| Nose (27-35) | 3-6px | 1.13px |
| Eyes (36-47) | 3-6px | 2.23px |
| Mouth (48-67) | 3-6px | 1.01px |

## PyMTCNN Analysis

PyMTCNN was checked using the same process:
- File sizes: Identical
- Diff output: 0 lines changed

**Conclusion**: PyMTCNN was properly preserved during repo split. No restoration needed.

## Checklist for Other Packages

- [ ] Find "max accuracy" commit
- [ ] Compare current vs expected performance
- [ ] Check if package was moved to separate repo
- [ ] Compare file sizes (old vs new)
- [ ] Diff critical files
- [ ] Look for model/algorithm changes
- [ ] Look for optimization changes
- [ ] Look for parameter changes
- [ ] Restore old files if needed
- [ ] Verify accuracy restored
- [ ] Check dependent packages
