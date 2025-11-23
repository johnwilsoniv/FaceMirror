#!/usr/bin/env python3
"""
Convert AU SVM models to CoreML for Apple Neural Engine acceleration.
Expected speedup: 2-3x for AU predictions on Apple Silicon.
"""

import numpy as np
import pickle
import struct
from pathlib import Path
import sys
import time
from typing import Dict, Tuple, List
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
import coremltools.proto.FeatureTypes_pb2 as ft

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


def read_svm_dat_file(filepath: str) -> Tuple[np.ndarray, float, Dict]:
    """
    Read SVM model from .dat file format used by OpenFace.

    Returns:
        weights: SVM weight vector
        bias: SVM bias term
        params: Additional SVM parameters
    """
    params = {}

    with open(filepath, 'rb') as f:
        # Read SVM type
        params['svm_type'] = struct.unpack('i', f.read(4))[0]

        # Read kernel type
        params['kernel_type'] = struct.unpack('i', f.read(4))[0]

        # Read dimensions
        params['total_sv'] = struct.unpack('i', f.read(4))[0]
        params['rho'] = struct.unpack('d', f.read(8))[0]  # Bias term

        # For linear SVMs, we can extract weights directly
        if params['kernel_type'] == 0:  # LINEAR kernel
            # Read number of features
            params['num_features'] = struct.unpack('i', f.read(4))[0]

            # Read weight vector
            weights = np.zeros(params['num_features'])
            for i in range(params['num_features']):
                weights[i] = struct.unpack('d', f.read(8))[0]

            return weights, -params['rho'], params
        else:
            raise ValueError(f"Non-linear kernel not supported for CoreML conversion")


def create_coreml_svm_classifier(
    weights: np.ndarray,
    bias: float,
    au_name: str,
    feature_dim: int = 5120
) -> ct.models.MLModel:
    """
    Create a CoreML model for SVM classification.

    Args:
        weights: SVM weight vector
        bias: SVM bias term
        au_name: Name of the AU (e.g., "AU_12")
        feature_dim: Input feature dimension

    Returns:
        CoreML model
    """
    from coremltools.models import MLModel
    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.models import datatypes

    # Create neural network builder (more flexible than SVM for our needs)
    builder = NeuralNetworkBuilder(
        input_features=[
            ("features", datatypes.Array(feature_dim))
        ],
        output_features=[
            (f"{au_name}_score", datatypes.Array(1)),
            (f"{au_name}_prob", datatypes.Array(1))
        ]
    )

    # Add inner product layer (equivalent to linear SVM)
    builder.add_inner_product(
        name="svm_decision",
        input_name="features",
        output_name="decision_value",
        input_channels=feature_dim,
        output_channels=1,
        W=weights.reshape(1, -1),  # Shape: (1, feature_dim)
        b=np.array([bias]),         # Shape: (1,)
        has_bias=True
    )

    # Add sigmoid activation for probability
    builder.add_activation(
        name="sigmoid",
        non_linearity="SIGMOID",
        input_name="decision_value",
        output_name=f"{au_name}_prob"
    )

    # Also output raw decision value
    builder.add_copy(
        name="copy_score",
        input_name="decision_value",
        output_name=f"{au_name}_score"
    )

    # Set metadata
    model = MLModel(builder.spec)
    model.short_description = f"AU {au_name} SVM Classifier"
    model.author = "Python AU Pipeline"
    model.version = "1.0"

    # Add metadata about compute units
    model.compute_unit = ct.ComputeUnit.ALL  # Use Neural Engine when available

    return model


def convert_all_au_svms(
    input_dir: str,
    output_dir: str,
    au_list: List[str] = None
) -> Dict[str, float]:
    """
    Convert all AU SVM models to CoreML format.

    Returns:
        Dictionary of conversion times
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default AU list if not provided
    if au_list is None:
        au_list = [
            "AU_01", "AU_02", "AU_04", "AU_05", "AU_06", "AU_07",
            "AU_09", "AU_10", "AU_12", "AU_14", "AU_15", "AU_17",
            "AU_20", "AU_23", "AU_25", "AU_26", "AU_28", "AU_45"
        ]

    conversion_times = {}
    successful_conversions = []
    failed_conversions = []

    print("=" * 60)
    print("AU SVM TO COREML CONVERSION")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Converting {len(au_list)} AU models...")
    print()

    for au in au_list:
        # Try both static and dynamic versions
        for model_type in ["static", "dynamic"]:
            model_name = f"{au}_{model_type}"
            dat_file = input_path / f"{model_name}.dat"

            if not dat_file.exists():
                continue

            try:
                print(f"Converting {model_name}...", end=" ")
                start = time.perf_counter()

                # Read SVM model
                weights, bias, params = read_svm_dat_file(str(dat_file))

                # Pad weights to expected feature dimension if needed
                feature_dim = 5120  # HOG (4096) + Geometry (1024)
                if len(weights) < feature_dim:
                    padded_weights = np.zeros(feature_dim)
                    padded_weights[:len(weights)] = weights
                    weights = padded_weights

                # Create CoreML model
                coreml_model = create_coreml_svm_classifier(
                    weights=weights,
                    bias=bias,
                    au_name=model_name,
                    feature_dim=feature_dim
                )

                # Save model
                output_file = output_path / f"{model_name}.mlmodel"
                coreml_model.save(str(output_file))

                elapsed = (time.perf_counter() - start) * 1000
                conversion_times[model_name] = elapsed
                successful_conversions.append(model_name)

                print(f"✓ ({elapsed:.1f}ms)")

            except Exception as e:
                failed_conversions.append(model_name)
                print(f"✗ Error: {str(e)}")

    # Summary
    print()
    print("=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Successful conversions: {len(successful_conversions)}")
    print(f"Failed conversions: {len(failed_conversions)}")

    if successful_conversions:
        avg_time = np.mean(list(conversion_times.values()))
        print(f"Average conversion time: {avg_time:.1f}ms")

        print("\nSuccessfully converted models:")
        for model in successful_conversions[:10]:
            print(f"  - {model}.mlmodel")
        if len(successful_conversions) > 10:
            print(f"  ... and {len(successful_conversions)-10} more")

    if failed_conversions:
        print("\nFailed conversions:")
        for model in failed_conversions:
            print(f"  - {model}")

    print("\nNext steps:")
    print("1. Test CoreML models for accuracy")
    print("2. Benchmark CoreML vs CPU performance")
    print("3. Integrate into production pipeline")

    return conversion_times


def test_coreml_model(model_path: str, test_features: np.ndarray) -> Tuple[float, float]:
    """
    Test a CoreML model with sample features.

    Returns:
        score: Raw SVM decision value
        probability: Sigmoid probability
    """
    import coremltools

    # Load model
    model = coremltools.models.MLModel(model_path)

    # Prepare input
    input_dict = {"features": test_features}

    # Run prediction
    start = time.perf_counter()
    output = model.predict(input_dict)
    inference_time = (time.perf_counter() - start) * 1000

    # Extract outputs
    score_key = [k for k in output.keys() if "_score" in k][0]
    prob_key = [k for k in output.keys() if "_prob" in k][0]

    score = output[score_key][0]
    prob = output[prob_key][0]

    print(f"  Inference time: {inference_time:.2f}ms")
    print(f"  Score: {score:.3f}")
    print(f"  Probability: {prob:.3f}")

    return score, prob


def benchmark_coreml_vs_numpy():
    """
    Benchmark CoreML vs NumPy for SVM inference.
    """
    print("\n" + "=" * 60)
    print("COREML VS NUMPY BENCHMARK")
    print("=" * 60)

    # Create dummy SVM model
    feature_dim = 5120
    weights = np.random.randn(feature_dim).astype(np.float32)
    bias = np.random.randn()
    features = np.random.randn(feature_dim).astype(np.float32)

    # NumPy benchmark
    n_iterations = 1000
    print(f"\nNumPy SVM ({n_iterations} iterations):")
    start = time.perf_counter()
    for _ in range(n_iterations):
        score = np.dot(weights, features) + bias
        prob = 1.0 / (1.0 + np.exp(-score))
    numpy_time = (time.perf_counter() - start) * 1000
    print(f"  Total time: {numpy_time:.1f}ms")
    print(f"  Per inference: {numpy_time/n_iterations:.3f}ms")

    # Create and test CoreML model
    print(f"\nCoreML SVM (first run includes compilation):")
    coreml_model = create_coreml_svm_classifier(
        weights=weights,
        bias=bias,
        au_name="test",
        feature_dim=feature_dim
    )

    # Save temporarily
    temp_path = "/tmp/test_au_svm.mlmodel"
    coreml_model.save(temp_path)

    # Benchmark CoreML
    model = ct.models.MLModel(temp_path)

    # First run (includes compilation)
    input_dict = {"features": features}
    start = time.perf_counter()
    _ = model.predict(input_dict)
    first_time = (time.perf_counter() - start) * 1000
    print(f"  First run: {first_time:.2f}ms")

    # Warm runs
    print(f"\nCoreML SVM ({n_iterations} iterations, warm):")
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = model.predict(input_dict)
    coreml_time = (time.perf_counter() - start) * 1000
    print(f"  Total time: {coreml_time:.1f}ms")
    print(f"  Per inference: {coreml_time/n_iterations:.3f}ms")

    print(f"\nSpeedup: {numpy_time/coreml_time:.1f}x")
    print("\nNote: CoreML uses Apple Neural Engine on M-series Macs")
    print("Expected speedup with batch processing: 2-3x")


if __name__ == "__main__":
    # Check if coremltools is installed
    try:
        import coremltools
        print(f"CoreML Tools version: {coremltools.__version__}")
    except ImportError:
        print("ERROR: coremltools not installed!")
        print("Install with: pip install coremltools")
        sys.exit(1)

    # Convert AU SVMs
    input_dir = "pyfaceau/weights/AU_predictors/svm_combined"
    output_dir = "pyfaceau/weights/AU_predictors/coreml_models"

    if Path(input_dir).exists():
        conversion_times = convert_all_au_svms(input_dir, output_dir)

        # Run benchmark
        benchmark_coreml_vs_numpy()
    else:
        print(f"Error: Input directory not found: {input_dir}")
        print("Please ensure you're running from the project root directory")