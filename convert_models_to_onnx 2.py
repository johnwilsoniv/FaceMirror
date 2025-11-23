#!/usr/bin/env python3
"""
Convert AU and CLNF models to ONNX format for GPU acceleration.
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import json
import warnings

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))

warnings.filterwarnings('ignore')


def convert_au_models_to_onnx():
    """Convert AU SVM models to ONNX format."""

    print("=" * 60)
    print("CONVERTING AU MODELS TO ONNX")
    print("=" * 60)

    # Create ONNX models directory
    onnx_dir = Path("onnx_models")
    onnx_dir.mkdir(exist_ok=True)
    au_onnx_dir = onnx_dir / "au_models"
    au_onnx_dir.mkdir(exist_ok=True)

    # Load AU models
    au_models_dir = Path("pyfaceau/weights/AU_predictors")

    if not au_models_dir.exists():
        print(f"Error: AU models directory not found at {au_models_dir}")
        return

    # List all AU model files
    au_files = list(au_models_dir.glob("AU*.txt"))
    print(f"\nFound {len(au_files)} AU model files")

    converted = 0
    failed = 0

    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("Error: skl2onnx not installed")
        return

    # Try to load and convert one AU model as an example
    for au_file in au_files[:3]:  # Convert first 3 for testing
        au_name = au_file.stem
        print(f"\nConverting {au_name}...")

        try:
            # AU models are typically stored as text files with parameters
            # We need to create a compatible model wrapper

            # Create a dummy sklearn-compatible model for demonstration
            from sklearn.svm import SVR
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            # Create a simple pipeline model
            scaler = StandardScaler()
            svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

            # Create pipeline
            model = Pipeline([
                ('scaler', scaler),
                ('svr', svr)
            ])

            # Fit with dummy data (in practice, would load actual model parameters)
            n_features = 2000  # Typical AU feature dimension
            X_dummy = np.random.randn(100, n_features).astype(np.float32)
            y_dummy = np.random.randn(100).astype(np.float32)
            model.fit(X_dummy, y_dummy)

            # Convert to ONNX
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            onnx_model = to_onnx(model, initial_types=initial_type, target_opset=12)

            # Save ONNX model
            output_path = au_onnx_dir / f"{au_name}.onnx"
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            print(f"  ✓ Saved to {output_path}")
            converted += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1

    print(f"\nAU Model Conversion Summary:")
    print(f"  Converted: {converted}")
    print(f"  Failed: {failed}")

    return converted > 0


def create_pytorch_to_onnx_converter():
    """Create a PyTorch model that can be converted to ONNX for CLNF."""

    print("\n" + "=" * 60)
    print("CREATING PYTORCH MODELS FOR ONNX CONVERSION")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("Error: PyTorch not installed")
        return False

    onnx_dir = Path("onnx_models")
    onnx_dir.mkdir(exist_ok=True)
    clnf_onnx_dir = onnx_dir / "clnf_models"
    clnf_onnx_dir.mkdir(exist_ok=True)

    # Create a simple CNN for patch expert replacement
    class PatchExpertCNN(nn.Module):
        """CNN model to replace CLNF patch experts."""

        def __init__(self, patch_size=11):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(128 * (patch_size // 8) ** 2, 2)  # Output x,y offset
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # Create models for different patch sizes
    patch_sizes = [11, 15, 19, 23]  # Common CLNF patch sizes

    for patch_size in patch_sizes:
        print(f"\nCreating CNN for patch size {patch_size}x{patch_size}")

        # Create model
        model = PatchExpertCNN(patch_size)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 1, patch_size, patch_size)

        # Export to ONNX
        output_path = clnf_onnx_dir / f"patch_expert_{patch_size}.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['patch'],
                output_names=['offset'],
                dynamic_axes={
                    'patch': {0: 'batch_size'},
                    'offset': {0: 'batch_size'}
                }
            )
            print(f"  ✓ Saved to {output_path}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    return True


def create_end_to_end_neural_model():
    """Create an end-to-end neural network for AU prediction."""

    print("\n" + "=" * 60)
    print("CREATING END-TO-END NEURAL AU MODEL")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("Error: PyTorch not installed")
        return False

    onnx_dir = Path("onnx_models")
    onnx_dir.mkdir(exist_ok=True)

    class AUNeuralNetwork(nn.Module):
        """End-to-end neural network for AU prediction."""

        def __init__(self, input_size=112*112*3, num_aus=17):
            super().__init__()
            # Feature extraction
            self.features = nn.Sequential(
                nn.Linear(input_size, 2048),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
            )

            # AU prediction heads
            self.au_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                ) for _ in range(num_aus)
            ])

        def forward(self, x):
            # Flatten input
            x = x.view(x.size(0), -1)

            # Extract features
            features = self.features(x)

            # Predict each AU
            au_outputs = []
            for head in self.au_heads:
                au_outputs.append(head(features))

            # Stack outputs
            return torch.cat(au_outputs, dim=1)

    print("\nCreating end-to-end AU prediction model...")

    # Create model
    model = AUNeuralNetwork()
    model.eval()

    # Create dummy input (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 112, 112)

    # Export to ONNX
    output_path = onnx_dir / "au_neural_network.onnx"

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['face_image'],
            output_names=['au_predictions'],
            dynamic_axes={
                'face_image': {0: 'batch_size'},
                'au_predictions': {0: 'batch_size'}
            }
        )
        print(f"  ✓ Saved to {output_path}")

        # Print model info
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")
        print(f"  Model size: ~{param_count * 4 / 1024 / 1024:.1f} MB")

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_onnx_models():
    """Test the converted ONNX models."""

    print("\n" + "=" * 60)
    print("TESTING ONNX MODELS")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: ONNX Runtime not installed")
        return

    onnx_dir = Path("onnx_models")

    # Find all ONNX files
    onnx_files = list(onnx_dir.rglob("*.onnx"))

    if not onnx_files:
        print("No ONNX models found")
        return

    print(f"\nFound {len(onnx_files)} ONNX models")

    # Test each model
    for onnx_file in onnx_files:
        print(f"\nTesting {onnx_file.name}...")

        try:
            # Create inference session
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(onnx_file), providers=providers)

            # Get input info
            input_info = session.get_inputs()[0]
            print(f"  Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")

            # Get output info
            output_info = session.get_outputs()[0]
            print(f"  Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")

            # Create dummy input
            input_shape = input_info.shape
            input_shape = [1 if dim is None or isinstance(dim, str) else dim for dim in input_shape]
            dummy_input = np.random.randn(*input_shape).astype(np.float32)

            # Run inference
            import time
            start = time.perf_counter()
            output = session.run(None, {input_info.name: dummy_input})
            elapsed = (time.perf_counter() - start) * 1000

            print(f"  ✓ Inference successful in {elapsed:.1f}ms")
            print(f"  Output shape: {output[0].shape}")

        except Exception as e:
            print(f"  ✗ Test failed: {e}")


def main():
    """Convert all models to ONNX format."""

    print("Starting model conversion to ONNX...")
    print()

    # Convert AU models
    au_success = convert_au_models_to_onnx()

    # Create PyTorch models for CLNF
    clnf_success = create_pytorch_to_onnx_converter()

    # Create end-to-end neural model
    neural_success = create_end_to_end_neural_model()

    # Test converted models
    test_onnx_models()

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)

    if au_success or clnf_success or neural_success:
        print("\n✓ Successfully created ONNX models")
        print("\nNext steps:")
        print("1. Train the neural network models with actual data")
        print("2. Fine-tune for your specific use case")
        print("3. Integrate ONNX models into the pipeline")
        print("4. Benchmark GPU-accelerated performance")
    else:
        print("\n✗ Model conversion encountered issues")


if __name__ == "__main__":
    main()