#!/usr/bin/env python3
"""Analyze PyFaceAU models for CoreML conversion"""

import onnx
from pathlib import Path

weights_dir = Path("pyfaceau/weights")

print("="*80)
print("PyFaceAU ONNX Models Analysis")
print("="*80)

# Analyze PFLD
pfld_path = weights_dir / "pfld_cunjian.onnx"
if pfld_path.exists():
    print("\n1. PFLD Landmark Detector")
    print(f"   Path: {pfld_path}")
    print(f"   Size: {pfld_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    model = onnx.load(str(pfld_path))
    print(f"   IR Version: {model.ir_version}")
    print(f"   Producer: {model.producer_name} {model.producer_version}")
    
    # Get input/output info
    graph = model.graph
    print(f"\n   Inputs:")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
        print(f"     - {inp.name}: {shape}")
    
    print(f"\n   Outputs:")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
        print(f"     - {out.name}: {shape}")
    
    print(f"\n   Operations: {len(graph.node)} nodes")
    
    # Check for dynamic shapes
    has_dynamic = any('dynamic' in str([d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]) for inp in graph.input)
    print(f"   Dynamic shapes: {has_dynamic}")

# Analyze RetinaFace
retinaface_path = weights_dir / "retinaface_mobilenet025_coreml.onnx"
if retinaface_path.exists():
    print("\n2. RetinaFace Detector")
    print(f"   Path: {retinaface_path}")
    print(f"   Size: {retinaface_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    model = onnx.load(str(retinaface_path))
    print(f"   IR Version: {model.ir_version}")
    print(f"   Producer: {model.producer_name} {model.producer_version}")
    
    graph = model.graph
    print(f"\n   Inputs:")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
        print(f"     - {inp.name}: {shape}")
    
    print(f"\n   Outputs:")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
        print(f"     - {out.name}: {shape}")
    
    print(f"\n   Operations: {len(graph.node)} nodes")

print("\n" + "="*80)
print("CoreML Conversion Feasibility")
print("="*80)

print("""
Benefits of Native CoreML:
  1. Better ANE (Apple Neural Engine) utilization
  2. Lower memory footprint
  3. Faster inference (typically 1.5-2x vs ONNX+CoreML)
  4. Consistency with PyMTCNN architecture
  5. Better power efficiency on Apple Silicon

Challenges:
  1. Need to validate numerical equivalence
  2. Maintain two model formats
  3. Handle any unsupported operations

Recommendation:
  ✓ Convert PFLD to CoreML (primary model, used for every frame)
  ? Convert RetinaFace (used less frequently, may not be worth it)
  
Next Steps:
  1. Convert PFLD ONNX → CoreML using coremltools
  2. Create PFLD backend classes (CoreMLPFLD, ONNXPFLD)
  3. Validate numerical equivalence
  4. Benchmark performance improvement
""")
