#!/usr/bin/env python3
"""
Inspect PNet ONNX model structure to identify potential export issues.
"""

import onnx
import numpy as np

def inspect_pnet_onnx():
    """
    Load and inspect PNet ONNX model structure.
    """
    model_path = "cpp_mtcnn_onnx/pnet.onnx"
    model = onnx.load(model_path)

    print(f"PNet ONNX Model Inspection")
    print(f"{'='*80}\n")

    # Model properties
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Model version: {model.model_version}")
    print(f"Doc string: {model.doc_string}\n")

    # Input/Output
    print(f"Inputs:")
    for input in model.graph.input:
        print(f"  {input.name}: {[dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in input.type.tensor_type.shape.dim]}")

    print(f"\nOutputs:")
    for output in model.graph.output:
        print(f"  {output.name}: {[dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in output.type.tensor_type.shape.dim]}")

    # Layers
    print(f"\n{'='*80}")
    print(f"Network Architecture:")
    print(f"{'='*80}\n")

    for i, node in enumerate(model.graph.node):
        print(f"Layer {i+1}: {node.op_type}")
        print(f"  Name: {node.name if node.name else '<unnamed>'}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")

        # Show attributes (important for checking activations, batch norm, etc.)
        if node.attribute:
            print(f"  Attributes:")
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    print(f"    {attr.name} = {attr.i}")
                elif attr.type == onnx.AttributeProto.FLOAT:
                    print(f"    {attr.name} = {attr.f}")
                elif attr.type == onnx.AttributeProto.INTS:
                    print(f"    {attr.name} = {list(attr.ints)}")
                elif attr.type == onnx.AttributeProto.FLOATS:
                    print(f"    {attr.name} = {list(attr.floats)[:5]}...")  # Truncate
                elif attr.type == onnx.AttributeProto.STRING:
                    print(f"    {attr.name} = {attr.s.decode('utf-8') if attr.s else '<empty>'}")

        print()

    # Initializers (weights)
    print(f"{'='*80}")
    print(f"Model Weights (Initializers):")
    print(f"{'='*80}\n")

    for init in model.graph.initializer:
        tensor = numpy_helper.to_array(init)
        print(f"{init.name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Type: {tensor.dtype}")
        print(f"  Stats: min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.mean():.6f}")
        print()

    # Look for potential issues
    print(f"{'='*80}")
    print(f"Potential Issues to Check:")
    print(f"{'='*80}\n")

    # Check for BatchNormalization
    has_batchnorm = any(node.op_type == 'BatchNormalization' for node in model.graph.node)
    print(f"1. BatchNormalization layers: {'YES' if has_batchnorm else 'NO'}")
    if has_batchnorm:
        print(f"   → Check if batch norm was fused or has wrong running stats")

    # Check for PReLU (MTCNN typically uses PReLU)
    has_prelu = any(node.op_type == 'PReLU' for node in model.graph.node)
    print(f"\n2. PReLU activation: {'YES' if has_prelu else 'NO'}")
    if not has_prelu:
        print(f"   → Check what activation is used instead (should be PReLU for MTCNN)")

    # Check final layer
    final_node = model.graph.node[-1]
    print(f"\n3. Final layer type: {final_node.op_type}")
    print(f"   → Should be Conv (no softmax in PNet output)")

    # Check for Transpose ops (could indicate channel ordering issues)
    transpose_nodes = [node for node in model.graph.node if node.op_type == 'Transpose']
    print(f"\n4. Transpose operations: {len(transpose_nodes)}")
    if transpose_nodes:
        print(f"   → Might indicate channel ordering issues")
        for node in transpose_nodes:
            print(f"      {node.name}: {node.input} → {node.output}")

if __name__ == "__main__":
    import onnx.numpy_helper as numpy_helper
    inspect_pnet_onnx()
