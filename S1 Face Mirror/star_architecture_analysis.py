#!/usr/bin/env python3
"""
STAR Model Architecture Analysis Script

This script analyzes the STAR landmark detection model to identify
operations that are incompatible with Apple Neural Engine.

Outputs a detailed report of:
- Layer types and counts
- Activation functions
- Normalization layers
- Dynamic operations
- Recommendations for optimization
"""

import torch
import pickle
from pathlib import Path
from collections import defaultdict
import sys

def analyze_model_structure(model):
    """
    Analyze model structure and identify problematic operations

    Returns:
        dict: Analysis results
    """
    analysis = {
        'total_layers': 0,
        'normalization_layers': defaultdict(list),
        'activation_functions': defaultdict(list),
        'convolution_layers': [],
        'linear_layers': [],
        'attention_layers': [],
        'dynamic_operations': [],
        'total_parameters': 0,
        'layer_details': []
    }

    # Count total parameters
    analysis['total_parameters'] = sum(p.numel() for p in model.parameters())

    # Analyze each module
    for name, module in model.named_modules():
        if len(name) == 0:  # Skip root module
            continue

        analysis['total_layers'] += 1
        module_type = type(module).__name__

        # Store layer details
        layer_info = {
            'name': name,
            'type': module_type,
            'ane_compatible': True,  # Assume compatible unless proven otherwise
            'reason': ''
        }

        # Check normalization layers (ANE incompatible: LayerNorm, InstanceNorm)
        if 'LayerNorm' in module_type:
            analysis['normalization_layers']['LayerNorm'].append(name)
            layer_info['ane_compatible'] = False
            layer_info['reason'] = 'LayerNorm causes partitions - replace with BatchNorm2d'
        elif 'InstanceNorm' in module_type:
            analysis['normalization_layers']['InstanceNorm'].append(name)
            layer_info['ane_compatible'] = False
            layer_info['reason'] = 'InstanceNorm not well supported - replace with BatchNorm2d'
        elif 'BatchNorm' in module_type:
            analysis['normalization_layers']['BatchNorm'].append(name)
            layer_info['reason'] = 'BatchNorm is ANE-compatible'

        # Check activation functions (ANE incompatible: GELU, SiLU, Sigmoid outside specific contexts)
        if 'GELU' in module_type:
            analysis['activation_functions']['GELU'].append(name)
            layer_info['ane_compatible'] = False
            layer_info['reason'] = 'GELU quantizes poorly - replace with ReLU6'
        elif 'SiLU' in module_type or 'Swish' in module_type:
            analysis['activation_functions']['SiLU'].append(name)
            layer_info['ane_compatible'] = False
            layer_info['reason'] = 'SiLU/Swish not ANE-compatible - replace with ReLU6'
        elif 'ReLU' in module_type:
            analysis['activation_functions']['ReLU'].append(name)
            layer_info['reason'] = 'ReLU/ReLU6 is ANE-compatible'
        elif 'Sigmoid' in module_type:
            analysis['activation_functions']['Sigmoid'].append(name)
            layer_info['ane_compatible'] = False
            layer_info['reason'] = 'Sigmoid can cause partitions - consider alternatives'
        elif 'Tanh' in module_type:
            analysis['activation_functions']['Tanh'].append(name)
            layer_info['ane_compatible'] = False
            layer_info['reason'] = 'Tanh can cause partitions - consider alternatives'

        # Check convolution layers (generally compatible)
        if 'Conv' in module_type:
            analysis['convolution_layers'].append(name)
            layer_info['reason'] = 'Convolution is ANE-compatible'

        # Check linear layers (may need conversion to Conv2d for better ANE support)
        if 'Linear' in module_type:
            analysis['linear_layers'].append(name)
            layer_info['ane_compatible'] = True  # Compatible but Conv2d is better
            layer_info['reason'] = 'Linear is compatible, but Conv2d 1x1 may be faster on ANE'

        # Check attention layers
        if 'Attention' in module_type or 'MultiheadAttention' in module_type:
            analysis['attention_layers'].append(name)
            layer_info['ane_compatible'] = False
            layer_info['reason'] = 'Attention likely uses dynamic operations - needs optimization'

        analysis['layer_details'].append(layer_info)

    return analysis


def print_analysis_report(analysis, output_file=None):
    """
    Print comprehensive analysis report

    Args:
        analysis: Analysis dict from analyze_model_structure
        output_file: Optional file path to save report
    """
    report_lines = []

    def log(line=''):
        """Helper to print and save to file"""
        print(line)
        report_lines.append(line)

    log("="*80)
    log("STAR MODEL ARCHITECTURE ANALYSIS")
    log("="*80)
    log()

    # Summary
    log("SUMMARY")
    log("-"*80)
    log(f"Total layers: {analysis['total_layers']}")
    log(f"Total parameters: {analysis['total_parameters']:,}")
    log(f"Model size (float32): ~{analysis['total_parameters'] * 4 / 1024 / 1024:.1f} MB")
    log()

    # Normalization layers
    log("NORMALIZATION LAYERS")
    log("-"*80)
    total_norm = sum(len(v) for v in analysis['normalization_layers'].values())
    if total_norm > 0:
        for norm_type, layers in analysis['normalization_layers'].items():
            status = "❌ INCOMPATIBLE" if norm_type in ['LayerNorm', 'InstanceNorm'] else "✅ COMPATIBLE"
            log(f"{norm_type}: {len(layers)} layers - {status}")
            if len(layers) > 0 and len(layers) <= 10:
                for layer in layers:
                    log(f"  - {layer}")
            elif len(layers) > 10:
                log(f"  - (showing first 5)")
                for layer in layers[:5]:
                    log(f"    - {layer}")
                log(f"  - ... and {len(layers)-5} more")
    else:
        log("No normalization layers found")
    log()

    # Activation functions
    log("ACTIVATION FUNCTIONS")
    log("-"*80)
    total_act = sum(len(v) for v in analysis['activation_functions'].values())
    if total_act > 0:
        for act_type, layers in analysis['activation_functions'].items():
            status = "✅ COMPATIBLE" if act_type in ['ReLU'] else "❌ INCOMPATIBLE"
            log(f"{act_type}: {len(layers)} layers - {status}")
            if len(layers) > 0 and len(layers) <= 10:
                for layer in layers:
                    log(f"  - {layer}")
            elif len(layers) > 10:
                log(f"  - (showing first 5)")
                for layer in layers[:5]:
                    log(f"    - {layer}")
                log(f"  - ... and {len(layers)-5} more")
    else:
        log("No activation functions found (likely in forward() method)")
    log()

    # Convolution layers
    log(f"CONVOLUTION LAYERS: {len(analysis['convolution_layers'])} - ✅ COMPATIBLE")
    log("-"*80)
    if len(analysis['convolution_layers']) > 0 and len(analysis['convolution_layers']) <= 10:
        for layer in analysis['convolution_layers']:
            log(f"  - {layer}")
    elif len(analysis['convolution_layers']) > 10:
        log(f"  Showing first 5 of {len(analysis['convolution_layers'])} layers:")
        for layer in analysis['convolution_layers'][:5]:
            log(f"  - {layer}")
    log()

    # Linear layers
    log(f"LINEAR LAYERS: {len(analysis['linear_layers'])} - ⚠️ CONSIDER CONVERTING TO CONV2D")
    log("-"*80)
    if len(analysis['linear_layers']) > 0 and len(analysis['linear_layers']) <= 10:
        for layer in analysis['linear_layers']:
            log(f"  - {layer}")
    elif len(analysis['linear_layers']) > 10:
        log(f"  Showing first 5 of {len(analysis['linear_layers'])} layers:")
        for layer in analysis['linear_layers'][:5]:
            log(f"  - {layer}")
    log()

    # Attention layers
    if len(analysis['attention_layers']) > 0:
        log(f"ATTENTION LAYERS: {len(analysis['attention_layers'])} - ❌ NEEDS OPTIMIZATION")
        log("-"*80)
        for layer in analysis['attention_layers']:
            log(f"  - {layer}")
        log()

    # Count incompatible layers
    incompatible_count = sum(1 for layer in analysis['layer_details'] if not layer['ane_compatible'])
    compatible_count = analysis['total_layers'] - incompatible_count

    log("ANE COMPATIBILITY SUMMARY")
    log("-"*80)
    log(f"✅ Compatible layers: {compatible_count}/{analysis['total_layers']} ({compatible_count/analysis['total_layers']*100:.1f}%)")
    log(f"❌ Incompatible layers: {incompatible_count}/{analysis['total_layers']} ({incompatible_count/analysis['total_layers']*100:.1f}%)")
    log()

    # Optimization recommendations
    log("OPTIMIZATION RECOMMENDATIONS")
    log("-"*80)

    layernorm_count = len(analysis['normalization_layers'].get('LayerNorm', []))
    if layernorm_count > 0:
        log(f"1. Replace {layernorm_count} LayerNorm layers with BatchNorm2d")
        log("   Priority: HIGH - LayerNorm causes graph partitions")

    gelu_count = len(analysis['activation_functions'].get('GELU', []))
    silu_count = len(analysis['activation_functions'].get('SiLU', []))
    if gelu_count > 0 or silu_count > 0:
        total_bad_acts = gelu_count + silu_count
        log(f"2. Replace {total_bad_acts} GELU/SiLU activations with ReLU6")
        log("   Priority: HIGH - Poor quantization and ANE compatibility")

    if len(analysis['attention_layers']) > 0:
        log(f"3. Optimize {len(analysis['attention_layers'])} attention mechanisms")
        log("   Priority: MEDIUM - Likely contains dynamic operations")
        log("   Recommendation: Replace nn.Linear with nn.Conv2d, remove dynamic reshapes")

    if len(analysis['linear_layers']) > 0:
        log(f"4. Consider converting {len(analysis['linear_layers'])} Linear layers to Conv2d (1x1)")
        log("   Priority: LOW - May improve ANE efficiency")

    log()
    log("="*80)
    log("NEXT STEPS")
    log("="*80)
    log("1. Create modified model with replacements (see PHASE2_IMPLEMENTATION_PLAN.md)")
    log("2. Fine-tune modified model for 1-2 epochs")
    log("3. Convert to CoreML with optimal settings")
    log("4. Analyze with Xcode Performance Report")
    log("5. Validate accuracy and benchmark performance")
    log()

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        log(f"Report saved to: {output_file}")


def main():
    """Main entry point"""
    print("STAR Model Architecture Analysis")
    print("="*80)
    print()

    # Locate model checkpoint
    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / 'weights' / 'Landmark_98.pkl'

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure the STAR model checkpoint is downloaded.")
        return 1

    print(f"Loading STAR checkpoint from: {checkpoint_path}")
    print()

    try:
        # Load model using OpenFace's LandmarkDetector
        # This handles all the complex pickle loading and model initialization
        print("Loading model using OpenFace LandmarkDetector...")
        print("(This may produce some logging output...)")
        print()

        from openface.landmark_detection import LandmarkDetector

        # CRITICAL: Patch STAR config BEFORE loading to prevent /work path errors
        # (Same approach as openface_integration.py)
        import openface.STAR.conf.alignment as alignment_conf
        import os

        original_init = alignment_conf.Alignment.__init__

        def patched_init(self_inner, args):
            original_init(self_inner, args)
            # Replace hardcoded /work path with a writable location
            import os.path as osp
            home_dir = os.path.expanduser('~')
            self_inner.ckpt_dir = os.path.join(home_dir, '.cache', 'openface', 'STAR')
            self_inner.work_dir = osp.join(self_inner.ckpt_dir, self_inner.data_definition, self_inner.folder)
            self_inner.model_dir = osp.join(self_inner.work_dir, 'model')
            self_inner.log_dir = osp.join(self_inner.work_dir, 'log')
            # Create directories if they don't exist
            os.makedirs(self_inner.ckpt_dir, exist_ok=True)

            # CRITICAL: Close and disable TensorBoard writer to prevent process hanging
            if hasattr(self_inner, 'writer') and self_inner.writer is not None:
                try:
                    self_inner.writer.close()
                except:
                    pass
            self_inner.writer = None

        alignment_conf.Alignment.__init__ = patched_init

        # Suppress verbose output during initialization
        import io
        import logging
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        logging.getLogger().setLevel(logging.CRITICAL)

        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            detector = LandmarkDetector(model_path=str(checkpoint_path), device='cpu')
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.getLogger().setLevel(logging.INFO)
            # Restore original __init__
            alignment_conf.Alignment.__init__ = original_init

        # The model is stored in detector.alignment.alignment
        model = detector.alignment.alignment

        print("✓ Successfully loaded STAR model via LandmarkDetector")

        # Ensure model is in eval mode
        model.eval()

        print()
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print()

        # Analyze the model
        print("Analyzing model architecture...")
        analysis = analyze_model_structure(model)

        print()

        # Print and save report
        output_file = script_dir / 'star_architecture_analysis_report.txt'
        print_analysis_report(analysis, output_file=output_file)

        return 0

    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
