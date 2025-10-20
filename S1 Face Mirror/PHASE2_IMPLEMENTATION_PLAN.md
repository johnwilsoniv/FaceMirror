# Phase 2: STAR Model Optimization - Implementation Plan

**Started:** 2025-10-17
**Expected Duration:** 3-4 weeks
**Expected Impact:** 3-4x speedup (~120s saved per session)

---

## Overview

**Current Performance (from profiling):**
- STAR inference: 29.7s total (49.6% of processing time)
- Average: 163.2ms per call
- 182 calls per video
- 82.6% ANE coverage, 18 partitions

**Target Performance:**
- STAR inference: ~10-13s total (3-4x speedup)
- Average: 50-60ms per call
- >95% ANE coverage, <5 partitions

**Estimated Savings:** ~17-20 seconds per video

---

## Current Status

### ✅ Assets Located

**Model Files:**
- ✅ STAR PyTorch checkpoint: `weights/Landmark_98.pkl` (52MB)
- ✅ Current ONNX model: `weights/star_landmark_98_coreml.onnx` (to be replaced)
- ✅ STAR implementation: Referenced in `openface_integration.py` and `onnx_star_detector.py`

### ⏳ Dependencies Needed

**Python Environment:**
- [ ] PyTorch (for loading/modifying model)
- [ ] CoreML Tools (`coremltools`) for conversion
- [ ] OpenFace 3.0 package (for STAR model definition)
- [ ] ONNX Runtime (for testing converted model)

**Development Tools:**
- [ ] Xcode (for Performance Report analysis)
- [ ] `powermetrics` (for ANE monitoring)

---

## Implementation Strategy (from STAR_OPTIMIZATION_GUIDE.md)

### Week 1-2: Model Architecture Modifications

#### Step 1: Set Up Environment & Load Model
- [ ] Install/verify PyTorch environment
- [ ] Install coremltools (`pip install coremltools`)
- [ ] Locate OpenFace 3.0 STAR model definition
- [ ] Load `Landmark_98.pkl` and inspect architecture
- [ ] Document current layer structure

#### Step 2: Analyze Current Architecture
- [ ] List all normalization layers (LayerNorm, InstanceNorm, BatchNorm)
- [ ] List all activation functions (GELU, SiLU, ReLU, etc.)
- [ ] Identify attention mechanism structure
- [ ] Document dynamic operations (reshapes, permutes)
- [ ] Map current 18 partitions to problematic operations

**Expected findings (from guide):**
- LayerNorm in transformer blocks (causes partitions)
- GELU activations (poor quantization)
- Dynamic reshapes in attention (CPU fallback)
- Multi-head attention with inefficient split/concat

#### Step 3: Replace Incompatible Operations

**Priority 1: Normalization Layers**
```python
# Replace all LayerNorm with BatchNorm2d
for name, module in model.named_modules():
    if isinstance(module, nn.LayerNorm):
        # Get dimensionality
        num_features = module.normalized_shape[0]
        # Create replacement
        bn = nn.BatchNorm2d(num_features)
        # Copy weights (approximate)
        if hasattr(module, 'weight'):
            bn.weight.data = module.weight.data
        if hasattr(module, 'bias'):
            bn.bias.data = module.bias.data
        # Replace in model
        setattr(parent, name, bn)
```

**Priority 2: Activation Functions**
```python
# Replace GELU/SiLU with ReLU6
for name, module in model.named_modules():
    if isinstance(module, (nn.GELU, nn.SiLU)):
        setattr(parent, name, nn.ReLU6())
```

**Priority 3: Attention Mechanism**
- Replace nn.Linear with nn.Conv2d (1x1 convolutions)
- Remove dynamic permute/transpose operations
- Keep tensors in (B, C, 1, S) format
- Use fixed sequence lengths

**Priority 4: Dynamic Reshapes**
- Use EnumeratedShapes during CoreML conversion
- Remove runtime shape queries
- Fix all tensor dimensions at conversion time

**Priority 5: Operation Fusion**
```python
# Fuse Conv + BN + Activation
model.eval()
model = torch.quantization.fuse_modules(
    model,
    [['conv', 'bn', 'relu']],
    inplace=True
)
```

#### Step 4: Fine-Tuning (Optional but Recommended)
- [ ] Load OpenFace training dataset (or subset)
- [ ] Fine-tune modified model for 1-2 epochs
- [ ] Validate landmark accuracy (<2% degradation target)
- [ ] Save modified checkpoint

**Note:** If no training data available, skip fine-tuning and validate accuracy after conversion.

---

### Week 2-3: CoreML Conversion & Optimization

#### Step 1: Prepare for Conversion
```python
import torch
import coremltools as ct

# Load modified model
model = load_modified_star_model()
model.eval()

# Convert to channels-last memory format (NHWC)
model = model.to(memory_format=torch.channels_last)

# Create example input
example_input = torch.randn(1, 3, 256, 256).to(memory_format=torch.channels_last)

# Trace model
traced_model = torch.jit.trace(model, example_input)
```

#### Step 2: Convert with Optimal Settings
```python
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(
        name="input_image",
        shape=ct.EnumeratedShapes(shapes=[
            (1, 3, 224, 224),
            (1, 3, 256, 256),
        ]),
        color_layout=ct.colorlayout.RGB,
        scale=1.0/255.0,  # Normalize during conversion
    )],
    convert_to="mlprogram",  # MLProgram format (iOS 15+)
    compute_precision=ct.precision.FLOAT16,  # FP16 for ANE
    compute_units=ct.ComputeUnit.CPU_AND_NE,  # Prefer ANE over GPU
    minimum_deployment_target=ct.target.iOS17,
    pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
)

coreml_model.save("star_landmark_98_optimized.mlpackage")
```

**Key Configuration Choices:**
1. **MLProgram** (not NeuralNetwork) - better FP16 support
2. **FLOAT16** - 2x memory reduction, 1.5-2x speedup, minimal accuracy loss
3. **CPU_AND_NE** - explicitly prefer ANE over GPU
4. **EnumeratedShapes** - fixed shapes compile to efficient code
5. **ImageType with preprocessing** - automatic normalization on ANE

#### Step 3: Analyze with Xcode
- [ ] Open `.mlpackage` in Xcode
- [ ] Product → Perform Action → Profile Model
- [ ] Check compute unit assignment per layer
- [ ] Count partitions (target: <5, currently: 18)
- [ ] Verify ANE coverage (target: >95%, currently: 82.6%)

#### Step 4: Iterate on Problematic Layers
- [ ] Identify remaining CPU/GPU operations
- [ ] Go back to architecture modifications if needed
- [ ] Re-convert and re-analyze
- [ ] Repeat until targets met

---

### Week 3-4: Export to ONNX & Integration

#### Step 1: Convert CoreML → ONNX
```python
# Export optimized CoreML model to ONNX
import onnxmltools
from onnxmltools.convert import convert_coreml

# Load CoreML model
coreml_model = ct.models.MLModel("star_landmark_98_optimized.mlpackage")

# Convert to ONNX
onnx_model = convert_coreml(coreml_model, "star_landmark_98_optimized")

# Save ONNX
onnx_model.save("weights/star_landmark_98_optimized.onnx")
```

#### Step 2: Test ONNX Model
- [ ] Load ONNX model in `onnx_star_detector.py`
- [ ] Run inference on test images
- [ ] Verify outputs match original (within tolerance)
- [ ] Benchmark inference time (target: 50-60ms)

#### Step 3: Integration
- [ ] Update `onnx_star_detector.py` to use new model
- [ ] Update model path in configuration
- [ ] Test full pipeline with Face Mirror
- [ ] Run profiler to confirm speedup

#### Step 4: Validation
- [ ] Compare landmark outputs with baseline (visual inspection)
- [ ] Measure landmark localization error (<2% increase target)
- [ ] Test on multiple videos
- [ ] Verify no crashes or errors

---

## Validation Checklist

### Accuracy Validation
- [ ] Landmark localization error: <2% increase from baseline
- [ ] Visual inspection: No artifacts in landmark placement
- [ ] Stable tracking: No jitter frame-to-frame
- [ ] Consistent across face sizes/orientations

### Performance Validation
- [ ] STAR inference: 50-60ms average on M1
- [ ] Pipeline FPS: ~4.8-5.0 FPS (up from 3.2 FPS)
- [ ] Memory usage: <2GB per process
- [ ] No thermal throttling during continuous processing
- [ ] ANE coverage: >95% (from 82.6%)
- [ ] Partitions: <5 (from 18)

### Quality Validation
- [ ] AU extraction quality unchanged
- [ ] No regression in face detection
- [ ] Downstream AU analysis produces valid results

---

## Troubleshooting Guide

### Issue: ANE Coverage Still Low (<90%)

**Debug Steps:**
1. Use Xcode Performance Report to identify CPU operations
2. Check for unsupported operations:
   - Tanh, Sigmoid in wrong context
   - Batch size > 1
   - Non-standard padding/pooling
   - Attention using einsum or bmm
3. Simplify attention to Conv2d + ReLU only

### Issue: Accuracy Degradation >2%

**Solutions:**
1. Fine-tune for more epochs (3-5 instead of 1-2)
2. Use mixed precision (FP16 most layers, FP32 critical layers)
3. Use knowledge distillation from original model
4. Reduce quantization aggressiveness

### Issue: Performance Not Improving

**Debug Steps:**
1. Verify ANE usage: `sudo powermetrics --samplers ane_power -i 1000`
2. Check compute unit configuration in code
3. Profile memory bandwidth (M1 Pro should show benefit)
4. Verify model actually using optimized version

### Issue: High Performance Variance

**Causes & Solutions:**
1. Thermal throttling → reduce CPU usage elsewhere
2. Cache misses → ensure model <12MB for M1 L3 cache
3. OS scheduling → set thread QoS to user-interactive
4. Background processes → test in isolated environment

---

## Files to Create/Modify

### New Files
- [ ] `convert_star_to_coreml.py` - Main conversion script
- [ ] `star_architecture_analysis.py` - Document current architecture
- [ ] `test_star_accuracy.py` - Validation script
- [ ] `benchmark_star.py` - Performance testing
- [ ] `weights/star_landmark_98_optimized.mlpackage` - Optimized CoreML model
- [ ] `weights/star_landmark_98_optimized.onnx` - Optimized ONNX model

### Modified Files
- [ ] `onnx_star_detector.py` - Update model path to use optimized version
- [ ] `OPTIMIZATION_IMPLEMENTATION_PLAN.md` - Update Phase 2 status

---

## Expected Outcomes

**Before Optimization:**
```
STAR_coreml:         29.7s (49.6% of total)
├─ Average:          163.2ms per call
├─ Calls:            182
├─ ANE Coverage:     82.6% (18 partitions)
└─ Baseline FPS:     3.2 FPS
```

**After Optimization:**
```
STAR_coreml:         ~10-13s (target, 3-4x speedup)
├─ Average:          50-60ms per call
├─ Calls:            182
├─ ANE Coverage:     >95% (<5 partitions)
└─ Pipeline FPS:     ~4.8-5.0 FPS (1.5-1.6x overall)
```

**Success Metrics:**
- ✅ 3-4x speedup on STAR inference
- ✅ <2% accuracy degradation
- ✅ >95% ANE coverage
- ✅ <5 CoreML partitions
- ✅ Pipeline FPS increases 1.5-1.6x

---

## Next Immediate Steps

1. **Verify Python environment**
   ```bash
   python3 -c "import torch; import coremltools; print('PyTorch:', torch.__version__); print('CoreML Tools:', coremltools.__version__)"
   ```

2. **Inspect Landmark_98.pkl**
   ```bash
   python3 -c "import pickle; model = pickle.load(open('weights/Landmark_98.pkl', 'rb')); print(type(model)); print(model.keys() if isinstance(model, dict) else dir(model))"
   ```

3. **Find OpenFace STAR model definition**
   - Check if OpenFace 3.0 is installed as package
   - Or locate source code in project
   - Or download from OpenFace 3.0 repository

4. **Create architecture analysis script**
   - Load model
   - Print layer structure
   - Identify problematic operations

---

**Last Updated:** 2025-10-17
**Status:** Planning Complete - Ready to Begin Implementation
**Next:** Verify environment and locate STAR model definition
