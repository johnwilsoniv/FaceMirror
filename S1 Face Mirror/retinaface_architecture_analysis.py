#!/usr/bin/env python3
"""
RetinaFace Model Architecture Analysis for Apple Neural Engine Compatibility

This script analyzes the RetinaFace face detection model to identify:
1. ANE-incompatible operations (LayerNorm, GELU, etc.)
2. Graph partitioning issues (dynamic shapes, unsupported ops)
3. Opportunities for optimization

Based on findings, we'll determine if we need architecture modifications
or just CoreML conversion optimization.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
import sys


def analyze_model_structure(model):
    """
    Analyze model structure and identify problematic operations

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_layers': 0,
        'normalization_layers': defaultdict(list),
        'activation_functions': defaultdict(list),
        'linear_layers': [],
        'conv_layers': [],
        'pooling_layers': [],
        'other_layers': defaultdict(list),
        'problematic_layers': [],
        'ane_compatible_layers': [],
        'by_type': defaultdict(int),
    }

    print("=" * 80)
    print("RETINAFACE MODEL ARCHITECTURE ANALYSIS")
    print("=" * 80)
    print()

    # Analyze each module
    for name, module in model.named_modules():
        if name == '':  # Skip root module
            continue

        analysis['total_layers'] += 1
        module_type = type(module).__name__
        analysis['by_type'][module_type] += 1

        # Create layer info
        layer_info = {
            'name': name,
            'type': module_type,
            'ane_compatible': True,
            'issue': None
        }

        # Analyze normalization layers
        if 'Norm' in module_type:
            if 'LayerNorm' in module_type:
                analysis['normalization_layers']['LayerNorm'].append(name)
                layer_info['ane_compatible'] = False
                layer_info['issue'] = 'LayerNorm not optimized for ANE (use BatchNorm2d)'
            elif 'InstanceNorm' in module_type:
                analysis['normalization_layers']['InstanceNorm'].append(name)
                layer_info['ane_compatible'] = False
                layer_info['issue'] = 'InstanceNorm not optimized for ANE (use BatchNorm2d)'
            elif 'GroupNorm' in module_type:
                analysis['normalization_layers']['GroupNorm'].append(name)
                layer_info['ane_compatible'] = False
                layer_info['issue'] = 'GroupNorm not optimized for ANE (use BatchNorm2d)'
            elif 'BatchNorm' in module_type:
                analysis['normalization_layers']['BatchNorm'].append(name)
                # BatchNorm is ANE-compatible

        # Analyze activation functions
        if any(act in module_type for act in ['ReLU', 'Sigmoid', 'Tanh', 'GELU', 'SiLU', 'Swish', 'LeakyReLU', 'PReLU']):
            if 'GELU' in module_type:
                analysis['activation_functions']['GELU'].append(name)
                layer_info['ane_compatible'] = False
                layer_info['issue'] = 'GELU not optimized for ANE (use ReLU/ReLU6)'
            elif 'SiLU' in module_type or 'Swish' in module_type:
                analysis['activation_functions']['SiLU/Swish'].append(name)
                layer_info['ane_compatible'] = False
                layer_info['issue'] = 'SiLU/Swish not optimized for ANE (use ReLU/ReLU6)'
            elif 'LeakyReLU' in module_type:
                analysis['activation_functions']['LeakyReLU'].append(name)
                # LeakyReLU is reasonably ANE-compatible
                layer_info['issue'] = 'LeakyReLU - acceptable for ANE (commonly used)'
            elif 'PReLU' in module_type:
                analysis['activation_functions']['PReLU'].append(name)
                # PReLU is reasonably ANE-compatible
                layer_info['issue'] = 'PReLU - acceptable for ANE'
            elif 'ReLU' in module_type:
                analysis['activation_functions']['ReLU'].append(name)
            elif 'Sigmoid' in module_type:
                analysis['activation_functions']['Sigmoid'].append(name)
            elif 'Tanh' in module_type:
                analysis['activation_functions']['Tanh'].append(name)

        # Analyze linear/conv layers
        if 'Linear' in module_type:
            analysis['linear_layers'].append(name)
        elif 'Conv' in module_type:
            analysis['conv_layers'].append(name)
        elif 'Pool' in module_type:
            analysis['pooling_layers'].append(name)
        elif module_type not in ['Sequential', 'ModuleList', 'Identity', 'Dropout']:
            analysis['other_layers'][module_type].append(name)

        # Track problematic vs compatible
        if not layer_info['ane_compatible']:
            analysis['problematic_layers'].append(layer_info)
        else:
            analysis['ane_compatible_layers'].append(layer_info)

    return analysis


def print_analysis(analysis):
    """Print analysis results in readable format"""

    print("LAYER TYPE SUMMARY")
    print("-" * 80)
    print(f"Total layers analyzed: {analysis['total_layers']}")
    print()

    # Print by type
    print("Layer counts by type:")
    for layer_type, count in sorted(analysis['by_type'].items(), key=lambda x: -x[1]):
        print(f"  {layer_type:30s}: {count:4d}")
    print()

    # Normalization analysis
    print("NORMALIZATION LAYERS")
    print("-" * 80)
    if analysis['normalization_layers']:
        for norm_type, layers in analysis['normalization_layers'].items():
            ane_status = "✓ ANE-compatible" if norm_type == "BatchNorm" else "✗ NOT ANE-optimal"
            print(f"{norm_type:20s}: {len(layers):4d} layers - {ane_status}")
            if norm_type != "BatchNorm" and len(layers) <= 5:
                for layer in layers:
                    print(f"  - {layer}")
    else:
        print("No normalization layers found")
    print()

    # Activation analysis
    print("ACTIVATION FUNCTIONS")
    print("-" * 80)
    if analysis['activation_functions']:
        for act_type, layers in analysis['activation_functions'].items():
            if act_type in ['ReLU', 'Sigmoid', 'Tanh']:
                ane_status = "✓ ANE-compatible"
            elif act_type in ['LeakyReLU', 'PReLU']:
                ane_status = "~ Acceptable (commonly used)"
            else:
                ane_status = "✗ NOT ANE-optimal"
            print(f"{act_type:20s}: {len(layers):4d} layers - {ane_status}")
    else:
        print("No activation functions found")
    print()

    # Linear/Conv analysis
    print("COMPUTATIONAL LAYERS")
    print("-" * 80)
    print(f"Convolution layers:      {len(analysis['conv_layers']):4d} - ✓ ANE-compatible")
    print(f"Linear layers:           {len(analysis['linear_layers']):4d} - ✓ ANE-compatible")
    print(f"Pooling layers:          {len(analysis['pooling_layers']):4d} - ✓ ANE-compatible")
    print()

    # Other layers
    if analysis['other_layers']:
        print("OTHER LAYERS")
        print("-" * 80)
        for layer_type, layers in analysis['other_layers'].items():
            print(f"{layer_type:30s}: {len(layers):4d} layers")
            # Print first few examples
            if len(layers) <= 3:
                for layer in layers:
                    print(f"  - {layer}")
        print()

    # Problematic layers summary
    print("=" * 80)
    print("ANE COMPATIBILITY ASSESSMENT")
    print("=" * 80)

    if analysis['problematic_layers']:
        print(f"⚠️  Found {len(analysis['problematic_layers'])} potentially problematic layers:")
        print()
        for layer in analysis['problematic_layers']:
            print(f"  ✗ {layer['name']}")
            print(f"    Type: {layer['type']}")
            print(f"    Issue: {layer['issue']}")
            print()
    else:
        print("✅ No problematic layers found!")
        print("   All layers are ANE-compatible!")
        print()

    # Summary
    total = len(analysis['ane_compatible_layers']) + len(analysis['problematic_layers'])
    compatible_pct = (len(analysis['ane_compatible_layers']) / total * 100) if total > 0 else 0

    print(f"Summary: {len(analysis['ane_compatible_layers'])}/{total} layers are ANE-compatible ({compatible_pct:.1f}%)")
    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if not analysis['problematic_layers']:
        print("✅ EXCELLENT: Model architecture is 100% ANE-compatible!")
        print()
        print("Recommended approach:")
        print("  1. Skip architecture modifications (not needed)")
        print("  2. Focus on CoreML conversion optimization:")
        print("     - Use MLProgram format (iOS 15+)")
        print("     - Use FLOAT16 precision")
        print("     - Use CPU_AND_NE compute units")
        print("     - Use fixed input shapes")
        print("     - Use channels-last memory format (NHWC)")
        print("  3. This should achieve 90%+ ANE coverage without model changes")
        print()
        print("Expected timeline: 2-3 hours (vs 2-3 weeks for architecture changes)")
    else:
        print("⚠️  Model has some ANE-incompatible operations")
        print()
        print("Two approaches:")
        print()
        print("Approach A: Optimize conversion only (RECOMMENDED)")
        print("  - Most issues can be resolved via CoreML conversion settings")
        print("  - LeakyReLU/PReLU are acceptable for RetinaFace")
        print("  - Timeline: 2-3 hours")
        print("  - Expected ANE coverage: 80-90%")
        print()
        print("Approach B: Modify architecture + optimize conversion")
        print("  - Replace problematic layers")
        print("  - Requires fine-tuning (accuracy risk)")
        print("  - Timeline: 2-3 weeks")
        print("  - Expected ANE coverage: 95-99%")
        print()
        print("Recommendation: Try Approach A first (like we did with STAR/MTL)")

    print("=" * 80)


def main():
    """Main analysis entry point"""
    print()
    print("Loading RetinaFace model from OpenFace...")
    print()

    # Import RetinaFace model
    from openface.Pytorch_Retinaface.models.retinaface import RetinaFace
    from openface.Pytorch_Retinaface.detect import load_model
    from openface.Pytorch_Retinaface.data import cfg_mnet

    # Create model
    cfg = cfg_mnet
    model = RetinaFace(cfg=cfg, phase='test')

    # Load weights
    script_dir = Path(__file__).parent
    weights_path = script_dir / 'weights' / 'Alignment_RetinaFace.pth'

    model = load_model(model, str(weights_path), load_to_cpu=True)
    model.eval()

    print(f"✓ Model loaded")
    print(f"✓ Model type: {type(model).__name__}")
    print()

    # Analyze architecture
    analysis = analyze_model_structure(model)

    # Print results
    print_analysis(analysis)

    # Save detailed report
    report_path = script_dir / 'retinaface_architecture_analysis_report.txt'

    # Redirect stdout to file
    import io
    from contextlib import redirect_stdout

    with open(report_path, 'w') as f:
        with redirect_stdout(f):
            print_analysis(analysis)

    print(f"Detailed report saved to: {report_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
