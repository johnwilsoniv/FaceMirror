# STAR Landmark Model Optimization Guide for Apple M1/M1 Pro

**Document Purpose:** This guide provides comprehensive instructions for optimizing the STAR Landmark-98 model for Apple Silicon (M1/M1 Pro) to achieve 3-4x performance improvement, reducing inference time from 174.5ms to 50-60ms per frame.

**Target Audience:** LLM or developer implementing CoreML optimizations for OpenFace 3.0 pipeline

**Status:** Based on production profiling data from OpenFace 3.0 face analysis pipeline

---

## Executive Summary

Performance profiling of the OpenFace 3.0 AU extraction pipeline revealed that **STAR Landmark-98 detection is the dominant bottleneck**, consuming 56% of total processing time at 174.5ms average per frame. This single model limits the entire pipeline to ~5.7 FPS. The model currently has 18 CoreML partitions with 82.6% Neural Engine coverage, indicating significant fragmentation causing expensive CPU↔ANE context switches.

**Primary Goal:** Optimize STAR model to achieve 50-60ms inference time (3-4x speedup) through partition reduction and improved Neural Engine utilization.

**Secondary Goal:** Fix RetinaFace postprocessing overhead (42.7ms → <5ms) for additional 10-15% pipeline speedup.

---

## Current Performance Profile

### Complete Pipeline Breakdown (323.988s total over 973 frames)

```
MODEL INFERENCE (93.5% of total = 302.900s):
├─ STAR_coreml:         169.823s (56.1%) ← PRIMARY BOTTLENECK
│  ├─ Count:            973 calls
│  ├─ Average:          174.5ms per call
│  ├─ Range:            85.6ms - 519.8ms
│  └─ ANE Coverage:     82.6% (18 partitions) ← NEEDS IMPROVEMENT
│
├─ MTL_coreml:          110.227s (36.4%)
│  ├─ Count:            1947 calls (2 per frame: left/right)
│  ├─ Average:          56.6ms per call
│  ├─ Range:            22.3ms - 569.7ms
│  └─ ANE Coverage:     69.0% (28 partitions)
│
└─ RetinaFace_coreml:   22.851s (7.5%)
   ├─ Count:            258 calls (every ~4 frames)
   ├─ Average:          88.6ms per call
   ├─ Range:            25.1ms - 723.1ms
   └─ ANE Coverage:     81.3% (3 partitions) ← Already good

POSTPROCESSING (3.5% of total = 11.433s):
├─ RetinaFace_postprocess: 11.020s (42.7ms avg) ← SECONDARY BOTTLENECK
├─ STAR_postprocess:        0.360s (0.4ms avg)
└─ MTL_postprocess:         0.053s (0.0ms avg)

PREPROCESSING (3.0% of total = 9.655s):
├─ MTL_preprocess:          5.067s (2.6ms avg)
├─ STAR_preprocess:         2.683s (2.8ms avg)
└─ RetinaFace_preprocess:   1.904s (7.4ms avg)
```

### Key Findings

1. **STAR dominates at 56% of processing time** - 174.5ms per frame limits pipeline to 5.7 FPS
2. **18 partitions indicate heavy fragmentation** - each partition boundary incurs CPU↔ANE transfer overhead
3. **82.6% ANE coverage is insufficient** - 17.4% CPU fallback at this scale creates significant bottleneck
4. **Wide performance variance** (85.6ms - 519.8ms) suggests thermal throttling or cache misses
5. **RetinaFace postprocessing is abnormally slow** at 42.7ms (should be <5ms)

---

## Apple M1 Architecture Context

### Hardware Specifications

**M1 Chip:**
- 4 performance cores (Firestorm, 3.2 GHz)
- 4 efficiency cores (Icestorm, 2.0 GHz)
- 8-core GPU (2.6 TFLOPS FP32)
- 16-core Neural Engine (11 TOPS)
- 70 GB/s unified memory bandwidth
- Shared L3 cache: 12-16 MB

**M1 Pro Chip:**
- 6-8 performance cores
- 2 efficiency cores
- 14-16 core GPU
- 16-core Neural Engine (11 TOPS, identical to M1)
- **200 GB/s unified memory bandwidth** (2.86x improvement)
- Larger L3 cache: 24 MB

### Neural Engine Characteristics

The Apple Neural Engine (ANE) is optimized for:
- **Single-inference low-latency execution** (not batch processing)
- **FP16 precision** operations
- **4D tensor format:** (Batch, Channels, 1, Sequence) with 64-byte aligned last axis
- **NHWC memory layout** (channels-last) for 20-30% better performance
- **Specific operator support:** Convolutions, activations (ReLU, ReLU6), normalization (BatchNorm)

**ANE Incompatibilities** (cause CPU fallback and partitions):
- Squeeze-and-Excitation blocks (global pooling + FC layers)
- Swish/SiLU activations (poor quantization, dynamic range -168 to 204)
- InstanceNorm and LayerNorm (use BatchNorm instead)
- Dynamic reshapes and control flow
- Depthwise convolutions with group size = 1 (low arithmetic intensity)
- Explicit padding operations (use Conv2d padding parameter)

---

## STAR Landmark Model Analysis

### Model Architecture

STAR (Style-based Transformer for Action Recognition) Landmark-98:
- **Purpose:** 98-point facial landmark detection for OpenFace 3.0
- **Input:** Face crop (typical size: 224×224 or 256×256)
- **Output:** 98 2D landmark coordinates
- **Parameters:** ~30-50M (estimated based on model size 52MB)
- **Current Performance:** 174.5ms per inference on M1

### Current CoreML Conversion Issues

```
Total operators:           855
Neural Engine operators:   706 (82.6%)
CPU fallback operators:    149 (17.4%)
CoreML partitions:         18 ← TOO MANY
```

**Partition Analysis:**
- 18 partitions means 18 separate subgraphs executed sequentially
- Each partition boundary requires:
  - Memory copy from ANE → CPU or CPU → ANE
  - Context switch overhead
  - Cache invalidation
- At 174ms total with 18 partitions: ~10ms overhead per partition = 180ms wasted

**Likely Culprits for Fragmentation:**
1. **Attention mechanisms** in transformer architecture (dynamic operations)
2. **Layer/Instance normalization** instead of BatchNorm
3. **GELU or other complex activations** instead of ReLU/ReLU6
4. **Dynamic reshapes** for sequence processing
5. **Positional encoding operations** not optimized for ANE
6. **Multi-head attention** with inefficient split/concat patterns

---

## Optimization Strategy: STAR Model

### Target Metrics

| Metric | Current | Target | Expected Speedup |
|--------|---------|--------|------------------|
| Inference Time | 174.5ms | 50-60ms | 3-4x |
| ANE Coverage | 82.6% | 95%+ | - |
| Partitions | 18 | <5 | - |
| FP16 Conversion | No (likely) | Yes | 1.5-2x |
| Memory Usage | ~52MB | ~26MB | 2x (with quantization) |

### Phase 1: Model Architecture Modifications (Highest Impact)

**Objective:** Reduce partitions from 18 to <5 by replacing ANE-incompatible operations

#### Step 1: Replace Normalization Layers

**Problem:** LayerNorm and InstanceNorm don't run on ANE
**Solution:** Replace with BatchNorm2d

```python
import torch
import torch.nn as nn

# Original (ANE-incompatible)
class StarBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # ← Replace this
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
    
    def forward(self, x):
        x = self.norm(x)
        return self.conv(x)

# Optimized (ANE-compatible)
class StarBlockOptimized(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)  # ← ANE-compatible
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
    
    def forward(self, x):
        x = self.norm(x)
        return self.conv(x)
```

**Implementation:**
1. Load original STAR PyTorch checkpoint
2. Identify all LayerNorm/InstanceNorm layers: `[name for name, m in model.named_modules() if isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d))]`
3. Replace with BatchNorm2d of same dimensionality
4. Fine-tune for 1-2 epochs on OpenFace training data to recover accuracy

**Expected Impact:** Reduce 4-6 partitions

#### Step 2: Replace Non-ReLU Activations

**Problem:** GELU, SiLU, Swish have poor quantization and may not run on ANE
**Solution:** Replace with ReLU6 (bounded output [0,6])

```python
# Original
self.activation = nn.GELU()

# Optimized
self.activation = nn.ReLU6()  # Bounded output helps quantization
```

**Implementation:**
1. Find all activation layers: `[name for name, m in model.named_modules() if 'gelu' in name.lower() or isinstance(m, nn.GELU)]`
2. Replace with `nn.ReLU6()` or `nn.ReLU()`
3. ReLU6 preferred over ReLU for mobile deployment (better quantization)

**Expected Impact:** Reduce 2-3 partitions, improve quantization accuracy by 20-30%

#### Step 3: Optimize Attention Mechanism

**Problem:** Multi-head attention uses dynamic operations (split, transpose, reshape) that fragment graph
**Solution:** Use ANE-friendly attention pattern from Apple's Transformer research

```python
# Original (ANE-unfriendly)
class MultiHeadAttention(nn.Module):
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # ← Dynamic permute causes partition
        q, k, v = qkv[0], qkv[1], qkv[2]
        # ... attention computation

# Optimized (ANE-friendly)
class MultiHeadAttentionOptimized(nn.Module):
    def forward(self, x):
        # Use separate conv layers instead of dynamic split
        q = self.q_conv(x)  # 1x1 conv, stays on ANE
        k = self.k_conv(x)
        v = self.v_conv(x)
        
        # Replace attention with 1x1 convolutions when possible
        # or use fixed reshapes with EnumeratedShapes
        attn = self.attention_conv(torch.cat([q, k, v], dim=1))
        return attn
```

**Apple's ANE Transformer Optimization Approach:**
1. Replace nn.Linear with nn.Conv2d (1x1 convolutions)
2. Keep tensors in (B, C, 1, S) format throughout
3. Use Conv2d operations instead of matrix multiplication
4. Avoid transpose and permute operations
5. Use fixed sequence lengths via EnumeratedShapes

**Expected Impact:** Reduce 6-8 partitions, 2-3x speedup on attention blocks

#### Step 4: Remove Dynamic Reshapes

**Problem:** Dynamic shape operations force CPU fallback
**Solution:** Use fixed shapes via CoreML EnumeratedShapes

```python
# During CoreML conversion
import coremltools as ct

# Instead of flexible shapes
model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 3, ct.RangeDim(), ct.RangeDim()))]  # ← BAD
)

# Use enumerated fixed shapes
model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(
        name="image",
        shape=ct.EnumeratedShapes(shapes=[
            (1, 3, 224, 224),
            (1, 3, 256, 256),
            (1, 3, 384, 384),
        ])
    )]  # ← GOOD
)
```

**Expected Impact:** Reduce 2-3 partitions

#### Step 5: Fuse Operations

**Problem:** Separate operations create partition boundaries
**Solution:** Fuse Conv → BatchNorm → Activation

```python
import torch.nn.utils.fusion as fusion

# Fuse Conv + BN
model = fusion.fuse_conv_bn_eval(model)

# Or manually during model definition
class FusedConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

**Expected Impact:** Reduce 1-2 partitions, 5-10% speedup

### Phase 2: CoreML Conversion Optimization

#### Conversion Configuration

```python
import coremltools as ct
import torch

# Load modified STAR model
model = load_modified_star_model()
model.eval()

# Convert to channels-last memory format (NHWC)
model = model.to(memory_format=torch.channels_last)

# Trace with example input
example_input = torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last)
traced_model = torch.jit.trace(model, example_input)

# Convert with optimal settings
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
    convert_to="mlprogram",  # Use MLProgram format (iOS 15+)
    compute_precision=ct.precision.FLOAT16,  # FP16 for ANE
    compute_units=ct.ComputeUnit.CPU_AND_NE,  # Prefer ANE over GPU
    minimum_deployment_target=ct.target.iOS17,  # Latest features
    
    # Enable optimizations
    pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
    
    # Skip model validation for faster conversion (validate manually)
    skip_model_load=False,
)

# Save model
coreml_model.save("star_landmark_optimized.mlpackage")
```

**Key Configuration Choices:**

1. **MLProgram format** (not NeuralNetwork):
   - Supports more operators on ANE
   - Better FP16 support
   - Required for iOS 15+

2. **FLOAT16 precision**:
   - ANE operates natively in FP16
   - 2x memory reduction
   - 1.5-2x speedup
   - Minimal accuracy loss (<1% for landmarks)

3. **CPU_AND_NE compute units** (not ALL):
   - Explicitly prefer ANE over GPU
   - ALL allows dynamic selection (unpredictable)
   - CPU_AND_NE forces ANE when compatible

4. **EnumeratedShapes** (not flexible shapes):
   - Fixed shapes compile to more efficient code
   - Avoids dynamic dispatching overhead
   - List all input sizes you'll use

5. **ImageType with color_layout**:
   - Automatic preprocessing on ANE
   - RGB normalization (scale=1.0/255.0)
   - Reduces preprocessing overhead

#### Post-Conversion Analysis

```python
import coremltools as ct

# Load converted model
model = ct.models.MLModel("star_landmark_optimized.mlpackage")

# Analyze compute unit assignment
spec = model.get_spec()

# Count operations by compute unit
ane_ops = 0
cpu_ops = 0
gpu_ops = 0

for op in spec.mlProgram.functions['main'].block_specializations.values():
    for block in op:
        for operation in block.operations:
            compute_unit = operation.attributes.get('compute_unit', 'CPU')
            if compute_unit == 'NeuralEngine':
                ane_ops += 1
            elif compute_unit == 'CPU':
                cpu_ops += 1
            elif compute_unit == 'GPU':
                gpu_ops += 1

print(f"ANE Operations: {ane_ops} ({ane_ops/(ane_ops+cpu_ops+gpu_ops)*100:.1f}%)")
print(f"CPU Operations: {cpu_ops} ({cpu_ops/(ane_ops+cpu_ops+gpu_ops)*100:.1f}%)")
print(f"GPU Operations: {gpu_ops} ({gpu_ops/(ane_ops+cpu_ops+gpu_ops)*100:.1f}%)")

# Target: >95% ANE operations
```

**Use Xcode Performance Report** for detailed analysis:
1. Open .mlpackage in Xcode
2. Product → Perform Action → Profile Model
3. View compute unit assignment per layer
4. Identify remaining CPU/GPU operations
5. Iterate on model modifications

### Phase 3: Quantization (Optional, for Further Optimization)

**Objective:** Reduce model size from 52MB → 13-26MB, improve inference speed by 1.5-2x

#### INT8 Quantization

```python
import coremltools as ct

# Load FP16 model
model = ct.models.MLModel("star_landmark_optimized.mlpackage")

# Define calibration data (representative inputs)
calibration_data = load_calibration_images(num_samples=256)

# Quantize to INT8
quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
    model,
    nbits=8,
    quantization_mode="linear",
    calibration_data=calibration_data,
)

quantized_model.save("star_landmark_int8.mlpackage")
```

#### Palettization (4-6 bits)

```python
import coremltools.optimize as cto

# Configure palettization
config = cto.coreml.OpPalettizerConfig(
    mode="kmeans",  # or "uniform"
    nbits=4,  # 4-bit palettization
    lut_function="linear",
)

# Apply palettization
palettized_model = cto.coreml.palettize_weights(
    model,
    config=config,
)

palettized_model.save("star_landmark_4bit.mlpackage")
```

**Quantization Accuracy Testing:**
- Test on validation set (1000+ images)
- Target: <2% landmark localization error increase
- If accuracy drops >2%, use mixed precision (FP16 for critical layers, INT8 for others)

### Phase 4: Memory Layout Optimization

#### Convert to NHWC (Channels-Last)

```python
import torch

# Load model
model = load_star_model()

# Convert to channels-last
model = model.to(memory_format=torch.channels_last)

# Also convert input during inference
input_tensor = input_tensor.to(memory_format=torch.channels_last)

# Trace and convert as usual
traced_model = torch.jit.trace(model, example_input)
```

**Benefits:**
- 20-30% faster convolution operations on ANE
- Better cache locality
- Reduced memory bandwidth by 30-40%
- 64-byte alignment benefits on last axis

**Verification:**
```python
# Check memory format
print(f"Model memory format: {next(model.parameters()).is_contiguous(memory_format=torch.channels_last)}")
```

---

## Optimization Strategy: RetinaFace Postprocessing

### Issue Analysis

**Current Performance:**
- Postprocessing: 42.7ms average (96.4% of total postprocessing time)
- Range: 11.1ms - 454.3ms (huge variance indicates inefficiency)
- Should be: <5ms for NMS + coordinate transform

**Likely Causes:**
1. **Inefficient NMS implementation** (Python loops instead of vectorized)
2. **Unnecessary CPU↔GPU transfers** (converting tensors repeatedly)
3. **Redundant coordinate transformations** (per-box operations)
4. **Unoptimized sorting** (Python sort instead of GPU/tensor operations)

### Solution: Vectorized NMS

```python
import torch
from torchvision.ops import nms, batched_nms
import numpy as np

# BEFORE (slow Python loop - 42ms)
def slow_nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: numpy array of shape (N, 4)
    scores: numpy array of shape (N,)
    """
    keep = []
    order = scores.argsort()[::-1]  # Python sort
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU with all remaining boxes (Python loop)
        ious = compute_iou(boxes[i], boxes[order[1:]])  # Slow!
        
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

# AFTER (vectorized Torch operations - <1ms)
def fast_nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: torch.Tensor of shape (N, 4) on GPU
    scores: torch.Tensor of shape (N,) on GPU
    """
    # Use torchvision's optimized NMS (CUDA/MPS accelerated)
    keep = nms(boxes, scores, iou_threshold)
    return keep

# Usage in postprocessing
def retinaface_postprocess_optimized(raw_output):
    # Keep tensors on GPU/MPS throughout
    boxes = torch.tensor(raw_output['boxes'], device='mps')  # or 'cuda'
    scores = torch.tensor(raw_output['scores'], device='mps')
    
    # Fast vectorized NMS
    keep_indices = fast_nms(boxes, scores, iou_threshold=0.5)
    
    # Index directly without CPU transfer
    final_boxes = boxes[keep_indices]
    final_scores = scores[keep_indices]
    
    # Only transfer final results to CPU
    return {
        'boxes': final_boxes.cpu().numpy(),
        'scores': final_scores.cpu().numpy()
    }
```

### Additional Optimizations

#### 1. Batch Coordinate Transforms

```python
# BEFORE (per-box loop - slow)
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    # Transform each box individually
    transformed[i] = transform_box(x1, y1, x2, y2)

# AFTER (vectorized - fast)
# Transform all boxes at once
x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
transformed = torch.stack([
    x1 * scale_x + offset_x,
    y1 * scale_y + offset_y,
    x2 * scale_x + offset_x,
    y2 * scale_y + offset_y,
], dim=1)
```

#### 2. Minimize Data Transfers

```python
# BEFORE (multiple CPU↔GPU transfers)
boxes_cpu = boxes.cpu().numpy()  # GPU → CPU
scores_cpu = scores.cpu().numpy()  # GPU → CPU
keep = nms(boxes_cpu, scores_cpu)  # CPU operation
result = boxes_cpu[keep]  # CPU operation
result_gpu = torch.tensor(result).to('mps')  # CPU → GPU

# AFTER (keep on GPU)
keep = nms(boxes, scores)  # GPU operation
result = boxes[keep]  # GPU operation
# Only transfer final result if needed
```

#### 3. Pre-allocate Buffers

```python
class RetinaFacePostprocessor:
    def __init__(self, max_detections=1000, device='mps'):
        self.device = device
        # Pre-allocate buffers
        self.boxes_buffer = torch.zeros((max_detections, 4), device=device)
        self.scores_buffer = torch.zeros(max_detections, device=device)
    
    def process(self, raw_output):
        # Reuse buffers instead of allocating each time
        n = len(raw_output['boxes'])
        self.boxes_buffer[:n] = torch.tensor(raw_output['boxes'], device=self.device)
        self.scores_buffer[:n] = torch.tensor(raw_output['scores'], device=self.device)
        
        # Process using pre-allocated buffers
        keep = nms(self.boxes_buffer[:n], self.scores_buffer[:n], 0.5)
        return keep
```

**Expected Impact:** Reduce RetinaFace postprocessing from 42.7ms → 2-5ms

---

## Implementation Roadmap

### Week 1: STAR Architecture Modifications (Days 1-5)

**Day 1-2: Analysis and Setup**
- [ ] Load STAR PyTorch checkpoint
- [ ] Document current architecture (layers, operations, shapes)
- [ ] Identify all LayerNorm/InstanceNorm layers
- [ ] Identify all non-ReLU activations (GELU, SiLU, etc.)
- [ ] Profile memory layout (NCHW vs NHWC)

**Day 3-4: Replace Incompatible Operations**
- [ ] Replace LayerNorm → BatchNorm2d (retain dimensionality)
- [ ] Replace GELU/SiLU → ReLU6
- [ ] Optimize attention mechanism (use Conv2d instead of Linear)
- [ ] Remove dynamic reshapes (use fixed shapes)
- [ ] Fuse Conv + BN + Activation layers

**Day 5: Fine-tuning and Validation**
- [ ] Fine-tune modified model on OpenFace training data (1-2 epochs)
- [ ] Validate landmark accuracy (target: <2% degradation)
- [ ] Save modified PyTorch checkpoint

### Week 2: CoreML Conversion and Optimization (Days 6-10)

**Day 6-7: Initial Conversion**
- [ ] Convert model to channels-last memory format
- [ ] Trace with torch.jit.trace
- [ ] Convert to CoreML with MLProgram format, FP16, CPU_AND_NE
- [ ] Use EnumeratedShapes for fixed input sizes
- [ ] Save initial CoreML model

**Day 8-9: Analysis and Iteration**
- [ ] Use Xcode Performance Report to analyze compute unit assignment
- [ ] Count partitions and ANE coverage (target: <5 partitions, >95% ANE)
- [ ] Identify remaining CPU/GPU operations
- [ ] Iterate on model modifications to eliminate remaining incompatibilities
- [ ] Re-convert and re-analyze

**Day 10: Validation and Benchmarking**
- [ ] Benchmark inference time on M1 (target: 50-60ms)
- [ ] Validate landmark accuracy on test set
- [ ] Profile memory usage
- [ ] Test on M1 Pro to verify performance gains

### Week 3: Quantization and Final Optimization (Days 11-15)

**Day 11-12: INT8 Quantization**
- [ ] Prepare calibration dataset (256+ representative images)
- [ ] Apply INT8 quantization
- [ ] Validate accuracy (target: <2% degradation)
- [ ] Benchmark performance (target: 35-45ms)

**Day 13: Palettization (Optional)**
- [ ] Apply 4-bit palettization
- [ ] Validate accuracy
- [ ] Benchmark performance and memory usage
- [ ] Choose best precision tradeoff

**Day 14-15: Integration and Testing**
- [ ] Integrate optimized STAR model into OpenFace pipeline
- [ ] Run full pipeline profiling
- [ ] Validate end-to-end performance
- [ ] Document performance improvements

### Week 4: RetinaFace Postprocessing Fix (Days 16-17)

**Day 16: NMS Optimization**
- [ ] Replace Python loop NMS with torchvision.ops.nms
- [ ] Keep tensors on GPU/MPS throughout postprocessing
- [ ] Vectorize coordinate transforms
- [ ] Pre-allocate buffers

**Day 17: Validation**
- [ ] Benchmark postprocessing time (target: <5ms)
- [ ] Validate detection accuracy unchanged
- [ ] Run full pipeline profiling
- [ ] Document final performance

---

## Performance Targets and Validation

### Target Metrics

| Component | Baseline | Target | Stretch Goal |
|-----------|----------|--------|--------------|
| **STAR Inference** | 174.5ms | 50-60ms | 40ms |
| **STAR ANE Coverage** | 82.6% | 95%+ | 98%+ |
| **STAR Partitions** | 18 | <5 | 1-3 |
| **RetinaFace Postprocess** | 42.7ms | <5ms | <2ms |
| **Total Pipeline** | 320ms (3.1 FPS) | 120-150ms (6.7-8.3 FPS) | 100ms (10 FPS) |

### Validation Checklist

**Accuracy Validation:**
- [ ] Landmark localization error: <2% increase from baseline
- [ ] Face detection recall: no degradation
- [ ] AU extraction accuracy: <1% degradation (downstream task)

**Performance Validation:**
- [ ] STAR inference: 50-60ms average on M1
- [ ] Pipeline FPS: 6.7-8.3 FPS sustained (target met)
- [ ] Memory usage: <2GB per process
- [ ] Thermal stability: no throttling during 5-minute continuous processing

**Quality Validation:**
- [ ] No visual artifacts in landmark detection
- [ ] Stable tracking (no jitter frame-to-frame)
- [ ] Consistent performance across face sizes and orientations

---

## Troubleshooting Guide

### Issue: ANE Coverage Still Low After Modifications

**Symptoms:**
- CoreML conversion shows <90% ANE operations
- Still seeing 10+ partitions

**Debug Steps:**
1. Use Xcode Performance Report to identify remaining CPU operations
2. Check for these common issues:
   - Unsupported activation functions (Tanh, Sigmoid in wrong context)
   - Batch size > 1 (ANE prefers batch=1)
   - Non-standard padding or pooling operations
   - Attention mechanisms using einsum or bmm
3. Simplify attention to use only Conv2d and ReLU

### Issue: Accuracy Degradation >2%

**Symptoms:**
- Landmark localization error increases significantly
- Visual artifacts in landmark placement

**Solutions:**
1. Use mixed precision: FP16 for most layers, FP32 for critical layers
2. Fine-tune for more epochs (3-5 instead of 1-2)
3. Use knowledge distillation from original model
4. Reduce quantization aggressiveness (INT8 → FP16, or 6-bit → 8-bit)

### Issue: Performance Not Improving

**Symptoms:**
- Inference time still >100ms after optimizations
- M1 Pro not faster than M1

**Debug Steps:**
1. Verify CoreML actually using ANE:
   ```bash
   sudo powermetrics --samplers ane_power -i 1000
   ```
   Should show ANE power usage during inference

2. Check compute unit assignment in code:
   ```python
   config = ct.models.MLModelConfiguration()
   config.computeUnits = ct.ComputeUnit.CPU_AND_NE  # Verify this
   model = ct.models.MLModel('model.mlpackage', configuration=config)
   ```

3. Profile memory bandwidth (if M1 Pro not faster, likely compute-bound not memory-bound)

### Issue: High Performance Variance

**Symptoms:**
- Inference time ranges from 50ms to 200ms
- Inconsistent FPS

**Causes and Solutions:**
1. **Thermal throttling:** Reduce CPU usage in other pipeline components
2. **Cache misses:** Ensure model fits in L3 cache (<12MB for M1)
3. **OS scheduling:** Set thread QoS to user-interactive priority
4. **Background processes:** Test in isolated environment

---

## Code Templates

### Complete STAR Conversion Pipeline

```python
import torch
import torch.nn as nn
import coremltools as ct
from typing import Optional

def optimize_star_for_coreml(
    original_checkpoint_path: str,
    output_path: str,
    input_size: tuple = (224, 224),
    fine_tune_epochs: int = 2,
    fine_tune_data: Optional[torch.utils.data.DataLoader] = None,
):
    """
    Complete pipeline to optimize STAR model for CoreML deployment on M1.
    
    Args:
        original_checkpoint_path: Path to original STAR PyTorch checkpoint
        output_path: Where to save optimized CoreML model
        input_size: Input image size (height, width)
        fine_tune_epochs: Number of fine-tuning epochs after modifications
        fine_tune_data: DataLoader with training data for fine-tuning
    """
    
    print("Step 1: Load original STAR model")
    model = load_star_model(original_checkpoint_path)
    model.eval()
    
    print("Step 2: Replace incompatible operations")
    model = replace_layernorm_with_batchnorm(model)
    model = replace_activations_with_relu6(model)
    model = optimize_attention_layers(model)
    model = fuse_conv_bn_relu(model)
    
    print("Step 3: Convert to channels-last memory format")
    model = model.to(memory_format=torch.channels_last)
    
    if fine_tune_data is not None:
        print(f"Step 4: Fine-tune for {fine_tune_epochs} epochs")
        model = fine_tune_model(model, fine_tune_data, epochs=fine_tune_epochs)
    
    print("Step 5: Trace model")
    example_input = torch.randn(1, 3, *input_size).to(memory_format=torch.channels_last)
    traced_model = torch.jit.trace(model, example_input)
    
    print("Step 6: Convert to CoreML")
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="input_image",
            shape=ct.EnumeratedShapes(shapes=[
                (1, 3, input_size[0], input_size[1]),
            ]),
            color_layout=ct.colorlayout.RGB,
            scale=1.0/255.0,
        )],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )
    
    print(f"Step 7: Save to {output_path}")
    coreml_model.save(output_path)
    
    print("Step 8: Analyze performance")
    analyze_coreml_model(output_path)
    
    return coreml_model

def replace_layernorm_with_batchnorm(model: nn.Module) -> nn.Module:
    """Replace all LayerNorm with BatchNorm2d"""
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            # Get normalized_shape (usually a single dimension)
            num_features = module.normalized_shape[0]
            bn = nn.BatchNorm2d(num_features)
            
            # Initialize with LayerNorm statistics (approximate)
            if hasattr(module, 'weight'):
                bn.weight.data = module.weight.data
            if hasattr(module, 'bias'):
                bn.bias.data = module.bias.data
            
            setattr(model, name, bn)
        else:
            # Recursively replace in submodules
            replace_layernorm_with_batchnorm(module)
    
    return model

def replace_activations_with_relu6(model: nn.Module) -> nn.Module:
    """Replace GELU, SiLU, Swish with ReLU6"""
    for name, module in model.named_children():
        if isinstance(module, (nn.GELU, nn.SiLU)):
            setattr(model, name, nn.ReLU6())
        else:
            replace_activations_with_relu6(module)
    
    return model

def optimize_attention_layers(model: nn.Module) -> nn.Module:
    """
    Replace attention mechanisms with ANE-friendly implementations.
    This is highly model-specific - adjust based on STAR architecture.
    """
    # TODO: Implement based on specific STAR architecture
    # General strategy:
    # 1. Replace nn.Linear with nn.Conv2d (1x1 convolutions)
    # 2. Remove dynamic reshapes/permutes
    # 3. Use fixed sequence lengths
    return model

def fuse_conv_bn_relu(model: nn.Module) -> nn.Module:
    """Fuse Conv + BatchNorm + ReLU for efficiency"""
    model.eval()  # Required for fusion
    model = torch.quantization.fuse_modules(
        model,
        [['conv', 'bn', 'relu']] * count_fusable_modules(model),
        inplace=True
    )
    return model

def analyze_coreml_model(model_path: str):
    """Analyze CoreML model compute unit assignment"""
    model = ct.models.MLModel(model_path)
    spec = model.get_spec()
    
    ane_ops = 0
    cpu_ops = 0
    total_ops = 0
    
    # Count operations by compute unit
    for func in spec.mlProgram.functions.values():
        for block_spec in func.block_specializations.values():
            for block in block_spec:
                for op in block.operations:
                    total_ops += 1
                    compute_unit = op.attributes.get('compute_unit', 'CPU')
                    if 'neural' in compute_unit.lower() or 'ane' in compute_unit.lower():
                        ane_ops += 1
                    else:
                        cpu_ops += 1
    
    ane_percentage = (ane_ops / total_ops * 100) if total_ops > 0 else 0
    
    print("\n" + "="*80)
    print("COREML MODEL ANALYSIS")
    print("="*80)
    print(f"Total Operations:     {total_ops}")
    print(f"ANE Operations:       {ane_ops} ({ane_percentage:.1f}%)")
    print(f"CPU Operations:       {cpu_ops} ({100-ane_percentage:.1f}%)")
    print(f"\nTarget: >95% ANE coverage")
    print(f"Status: {'✓ PASS' if ane_percentage > 95 else '✗ NEEDS IMPROVEMENT'}")
    print("="*80 + "\n")
```

### Fast NMS Implementation

```python
import torch
from torchvision.ops import nms

class FastRetinaFacePostprocessor:
    """Optimized RetinaFace postprocessing with fast NMS"""
    
    def __init__(
        self,
        max_detections: int = 1000,
        nms_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        device: str = 'mps',  # or 'cuda' or 'cpu'
    ):
        self.max_detections = max_detections
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Pre-allocate buffers
        self.boxes_buffer = torch.zeros((max_detections, 4), device=device)
        self.scores_buffer = torch.zeros(max_detections, device=device)
    
    def __call__(self, raw_output: dict) -> dict:
        """
        Process RetinaFace raw output with optimized NMS.
        
        Args:
            raw_output: dict with keys 'boxes' (Nx4), 'scores' (N,), 'landmarks' (Nx10)
        
        Returns:
            dict with filtered boxes, scores, landmarks
        """
        # Convert to tensors on device (avoid CPU↔GPU transfers)
        boxes = torch.as_tensor(raw_output['boxes'], dtype=torch.float32, device=self.device)
        scores = torch.as_tensor(raw_output['scores'], dtype=torch.float32, device=self.device)
        landmarks = torch.as_tensor(raw_output['landmarks'], dtype=torch.float32, device=self.device)
        
        # Filter by confidence threshold (vectorized)
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]
        
        if len(boxes) == 0:
            return {'boxes': [], 'scores': [], 'landmarks': []}
        
        # Fast NMS (GPU/MPS accelerated)
        keep_indices = nms(boxes, scores, self.nms_threshold)
        
        # Filter results
        final_boxes = boxes[keep_indices]
        final_scores = scores[keep_indices]
        final_landmarks = landmarks[keep_indices]
        
        # Only transfer to CPU at the very end
        return {
            'boxes': final_boxes.cpu().numpy(),
            'scores': final_scores.cpu().numpy(),
            'landmarks': final_landmarks.cpu().numpy(),
        }
```

---

## Expected Results

### Performance Improvements

**STAR Landmark Model:**
```
Before optimization:
- Inference time: 174.5ms
- ANE coverage: 82.6%
- Partitions: 18
- FPS contribution: ~5.7 FPS cap

After optimization:
- Inference time: 50-60ms (3x improvement)
- ANE coverage: 95%+
- Partitions: <5
- FPS contribution: ~16-20 FPS cap
```

**RetinaFace Postprocessing:**
```
Before optimization:
- Postprocessing time: 42.7ms
- Implementation: Python loops

After optimization:
- Postprocessing time: <5ms (8-10x improvement)
- Implementation: Vectorized GPU/MPS
```

**Complete Pipeline:**
```
Before optimization:
- Total time per frame: ~320ms
- FPS: 3.1 FPS
- Bottleneck: STAR (56% of time)

After optimization:
- Total time per frame: ~120-150ms
- FPS: 6.7-8.3 FPS (2.2-2.7x improvement)
- More balanced pipeline
```

### Success Criteria

**Must Have (Required for Success):**
- ✅ STAR inference: <60ms average
- ✅ ANE coverage: >90%
- ✅ Partitions: <8
- ✅ Landmark accuracy: <2% degradation
- ✅ Pipeline FPS: >6 FPS sustained

**Should Have (Target Goals):**
- ⭐ STAR inference: <55ms average
- ⭐ ANE coverage: >95%
- ⭐ Partitions: <5
- ⭐ Landmark accuracy: <1% degradation
- ⭐ Pipeline FPS: >7 FPS sustained

**Nice to Have (Stretch Goals):**
- 🎯 STAR inference: <45ms average
- 🎯 ANE coverage: >98%
- 🎯 Partitions: 1-3
- 🎯 Landmark accuracy: <0.5% degradation
- 🎯 Pipeline FPS: >8 FPS sustained

---

## References and Resources

### Apple Documentation

1. **Core ML Tools Documentation**
   - https://apple.github.io/coremltools/
   - PyTorch conversion workflow
   - Optimization techniques (pruning, quantization, palettization)

2. **Apple Machine Learning Research**
   - "Deploying Transformers on the Apple Neural Engine"
   - https://machinelearning.apple.com/research/neural-engine-transformers
   - Key insights on ANE-friendly architectures

3. **WWDC 2024: Bring Your ML Models to Apple Silicon**
   - https://developer.apple.com/videos/play/wwdc2024/10159/
   - Latest CoreML features and optimization techniques

### Community Resources

4. **Hugging Face: Core ML Optimization**
   - https://huggingface.co/docs/diffusers/en/optimization/coreml
   - Real-world examples of Stable Diffusion optimization

5. **Neural Engine Documentation (hollance/neural-engine)**
   - https://github.com/hollance/neural-engine
   - Unofficial documentation of ANE capabilities and limitations

### Research Papers

6. **EfficientNet-Lite (TensorFlow Blog)**
   - https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html
   - Mobile optimization strategies applicable to STAR

7. **OpenFace 3.0 Paper**
   - arXiv:2506.02891
   - Original architecture and design decisions

---

## Appendix: Profiling Commands

### Monitor Neural Engine Usage

```bash
# Real-time ANE power monitoring
sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 1000

# View ANE activity (requires asitop tool)
sudo asitop
```

### Xcode Profiling

```bash
# Profile CoreML model in Xcode
# 1. Open .mlpackage in Xcode
# 2. Product → Perform Action → Profile Model
# 3. Select M1 Mac as target device
# 4. View Performance Report
```

### Python Profiling

```python
import time
import numpy as np

def profile_model(model, input_data, num_iterations=100):
    """Profile CoreML model inference time"""
    times = []
    
    # Warmup
    for _ in range(10):
        _ = model.predict(input_data)
    
    # Actual profiling
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model.predict(input_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    print(f"Mean: {times.mean():.2f}ms")
    print(f"Std:  {times.std():.2f}ms")
    print(f"Min:  {times.min():.2f}ms")
    print(f"Max:  {times.max():.2f}ms")
    print(f"P50:  {np.percentile(times, 50):.2f}ms")
    print(f"P95:  {np.percentile(times, 95):.2f}ms")
    print(f"P99:  {np.percentile(times, 99):.2f}ms")
```

---

## Contact and Support

For questions or issues during implementation:
1. Review Apple's CoreML documentation first
2. Check Xcode Performance Report for compute unit assignment
3. Profile with powermetrics to verify ANE usage
4. Compare against baseline metrics in this document

**Document Version:** 1.0
**Last Updated:** 2025-10-17
**Target Hardware:** Apple M1, M1 Pro
**Target Framework:** CoreML 8.3+, iOS 17+, macOS 14+

---

END OF OPTIMIZATION GUIDE
