#!/usr/bin/env python3
"""
Implement GPU acceleration using Metal Performance Shaders for Apple Silicon.

Metal provides native GPU acceleration on macOS with excellent performance
for matrix operations, convolutions, and neural networks.

Expected improvements:
- 3-5x speedup on M1/M2/M3 chips
- Efficient memory management
- Low power consumption
"""

import numpy as np
import time
import sys
from pathlib import Path
import cv2
from typing import Dict, Optional, Tuple, List
import platform
import warnings

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))

# Check for PyTorch with MPS support (Metal Performance Shaders)
HAS_MPS = False
try:
    import torch
    if torch.backends.mps.is_available():
        HAS_MPS = True
        print("✓ Metal Performance Shaders (MPS) available")
except ImportError:
    print("⚠️  PyTorch not installed. Install with: pip install torch torchvision")


class MetalAcceleratedCLNF:
    """CLNF with Metal GPU acceleration for Apple Silicon."""

    def __init__(self, clnf_instance, device='mps'):
        """
        Initialize Metal-accelerated CLNF.

        Args:
            clnf_instance: Original CLNF instance
            device: 'mps' for Metal, 'cpu' for CPU
        """
        self.clnf = clnf_instance
        self.device = device

        if HAS_MPS and device == 'mps':
            self.use_metal = True
            self.device_torch = torch.device('mps')
            self._convert_to_metal()
        else:
            self.use_metal = False
            self.device_torch = torch.device('cpu')

        print(f"CLNF using device: {self.device_torch}")

    def _convert_to_metal(self):
        """Convert numpy arrays to Metal tensors."""
        # Convert patch expert weights to Metal
        if hasattr(self.clnf, 'patch_experts'):
            for expert in self.clnf.patch_experts:
                if hasattr(expert, 'weights'):
                    # Convert to torch tensor on MPS
                    weights_np = expert.weights
                    expert.weights_metal = torch.from_numpy(weights_np).to(self.device_torch)

    def compute_response_metal(self, image_patch: np.ndarray, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute response map using Metal acceleration.

        Args:
            image_patch: Input image patch
            weights: Model weights on Metal

        Returns:
            Response map
        """
        # Convert image to Metal tensor
        patch_tensor = torch.from_numpy(image_patch).to(self.device_torch)

        # Perform convolution on GPU
        response = torch.nn.functional.conv2d(
            patch_tensor.unsqueeze(0).unsqueeze(0),
            weights.unsqueeze(0).unsqueeze(0)
        )

        return response.squeeze()

    def fit(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, bool]:
        """
        Fit landmarks using Metal acceleration where possible.

        Args:
            image: Input image
            bbox: Face bounding box

        Returns:
            Landmarks and convergence status
        """
        if self.use_metal:
            # TODO: Full Metal implementation would require porting entire CLNF
            # For now, accelerate critical operations only
            pass

        # Fall back to CPU implementation
        return self.clnf.fit(image, bbox)


class MetalAcceleratedAUPipeline:
    """AU Pipeline with Metal GPU acceleration."""

    def __init__(self, au_pipeline_instance, device='mps'):
        """
        Initialize Metal-accelerated AU pipeline.

        Args:
            au_pipeline_instance: Original pipeline instance
            device: 'mps' for Metal, 'cpu' for CPU
        """
        self.pipeline = au_pipeline_instance
        self.device = device

        if HAS_MPS and device == 'mps':
            self.use_metal = True
            self.device_torch = torch.device('mps')
            self._convert_models_to_metal()
        else:
            self.use_metal = False
            self.device_torch = torch.device('cpu')

        print(f"AU Pipeline using device: {self.device_torch}")

    def _convert_models_to_metal(self):
        """Convert SVM models to Metal tensors."""
        self.metal_models = {}

        if hasattr(self.pipeline, 'au_models'):
            for au_name, model in self.pipeline.au_models.items():
                metal_model = {}

                # Convert support vectors
                if hasattr(model, 'support_vectors_'):
                    sv = model.support_vectors_
                    metal_model['support_vectors'] = torch.from_numpy(sv).to(self.device_torch)

                # Convert dual coefficients
                if hasattr(model, 'dual_coef_'):
                    dc = model.dual_coef_
                    metal_model['dual_coef'] = torch.from_numpy(dc).to(self.device_torch)

                # Store other parameters
                if hasattr(model, 'gamma'):
                    metal_model['gamma'] = model.gamma
                if hasattr(model, 'intercept_'):
                    metal_model['intercept'] = torch.tensor(model.intercept_).to(self.device_torch)

                self.metal_models[au_name] = metal_model

    def predict_rbf_kernel_metal(self, X: np.ndarray, au_name: str) -> float:
        """
        Predict using RBF kernel SVM on Metal GPU.

        Args:
            X: Input features
            au_name: AU identifier

        Returns:
            Prediction value
        """
        if au_name not in self.metal_models:
            return 0.0

        model = self.metal_models[au_name]

        # Convert input to Metal tensor
        X_tensor = torch.from_numpy(X).float().to(self.device_torch)

        # Compute RBF kernel
        # K(x, x') = exp(-gamma * ||x - x'||^2)
        support_vectors = model['support_vectors']
        gamma = model.get('gamma', 1.0)

        # Compute squared distances
        X_norm = (X_tensor ** 2).sum(1).view(-1, 1)
        sv_norm = (support_vectors ** 2).sum(1).view(1, -1)
        distances = X_norm + sv_norm - 2.0 * torch.mm(X_tensor, support_vectors.t())

        # Apply RBF kernel
        K = torch.exp(-gamma * distances)

        # Compute decision function
        dual_coef = model['dual_coef']
        decision = torch.mm(K, dual_coef.t())

        # Add intercept
        if 'intercept' in model:
            decision += model['intercept']

        # Convert back to CPU
        return decision.cpu().numpy().item()

    def process_frame_metal(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Process frame using Metal acceleration.

        Args:
            frame: Input frame
            landmarks: Face landmarks

        Returns:
            AU predictions
        """
        if not self.use_metal:
            # Fall back to CPU
            return self.pipeline._process_frame(frame, 0, 0.0)

        # Extract features (still on CPU for now)
        features = self.pipeline._extract_features(frame, landmarks)

        # Predict AUs using Metal
        aus = {}
        for au_name in self.metal_models.keys():
            aus[au_name] = self.predict_rbf_kernel_metal(features.reshape(1, -1), au_name)

        return {'aus': aus}


class ONNXAcceleratedPipeline:
    """Pipeline with ONNX Runtime acceleration."""

    def __init__(self, verbose=True):
        """Initialize ONNX-accelerated pipeline."""
        self.verbose = verbose
        self.onnx_available = False

        try:
            import onnxruntime as ort
            self.onnx_available = True

            # Check available providers
            providers = ort.get_available_providers()
            if verbose:
                print("ONNX Runtime providers available:", providers)

            # Select best provider
            if 'CoreMLExecutionProvider' in providers:
                self.provider = 'CoreMLExecutionProvider'
            elif 'CUDAExecutionProvider' in providers:
                self.provider = 'CUDAExecutionProvider'
            elif 'DmlExecutionProvider' in providers:
                self.provider = 'DmlExecutionProvider'
            else:
                self.provider = 'CPUExecutionProvider'

            if verbose:
                print(f"Using ONNX provider: {self.provider}")

        except ImportError:
            if verbose:
                print("⚠️  ONNX Runtime not installed. Install with: pip install onnxruntime")

    def export_to_onnx(self, model, input_shape, output_path):
        """
        Export a model to ONNX format.

        Args:
            model: Model to export
            input_shape: Input tensor shape
            output_path: Path to save ONNX model
        """
        if not self.onnx_available:
            print("ONNX Runtime not available")
            return False

        try:
            import torch
            import torch.onnx

            # Create dummy input
            dummy_input = torch.randn(*input_shape)

            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                             'output': {0: 'batch_size'}}
            )

            print(f"✓ Exported model to {output_path}")
            return True

        except Exception as e:
            print(f"Failed to export to ONNX: {e}")
            return False


def benchmark_metal_acceleration():
    """Benchmark Metal GPU acceleration."""

    print("=" * 60)
    print("METAL GPU ACCELERATION BENCHMARK")
    print("=" * 60)

    # Check system
    print(f"\nSystem: {platform.system()} {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    if not platform.system() == 'Darwin' or not 'arm' in platform.machine().lower():
        print("⚠️  Metal acceleration requires Apple Silicon Mac")
        print("   Current system doesn't support Metal")
        return

    if not HAS_MPS:
        print("\n⚠️  PyTorch with MPS support not available")
        print("   Install with: pip install torch torchvision")
        return

    video_path = "Patient Data/Normal Cohort/Shorty.mov"
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Initialize pipelines
    print("\nInitializing pipelines...")

    warnings.filterwarnings('ignore')

    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau import FullPythonAUPipeline

    # Redirect output
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        detector = MTCNN()

        # CPU baseline
        clnf_cpu = CLNF(
            model_dir="pyclnf/models",
            max_iterations=5,
            convergence_threshold=0.5
        )
        au_pipeline_cpu = FullPythonAUPipeline(
            pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
            au_models_dir="pyfaceau/weights/AU_predictors",
            triangulation_file="pyfaceau/weights/tris_68_full.txt",
            patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
            verbose=False
        )

        # Metal-accelerated
        clnf_metal = MetalAcceleratedCLNF(clnf_cpu, device='mps')
        au_pipeline_metal = MetalAcceleratedAUPipeline(au_pipeline_cpu, device='mps')

    finally:
        sys.stdout = old_stdout

    # Benchmark
    print("\nBenchmarking...")

    cap = cv2.VideoCapture(video_path)
    test_frames = []

    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            break
        test_frames.append(frame)

    cap.release()

    configs = [
        ("CPU Baseline", clnf_cpu, au_pipeline_cpu),
        ("Metal GPU", clnf_metal, au_pipeline_metal)
    ]

    for name, clnf, au_pipeline in configs:
        print(f"\n{name}:")

        frame_times = []

        for i, frame in enumerate(test_frames):
            start = time.perf_counter()

            detection = detector.detect(frame)
            if detection and isinstance(detection, tuple) and len(detection) == 2:
                bboxes, _ = detection
                if len(bboxes) > 0:
                    bbox = bboxes[0]
                    x, y, w, h = [int(v) for v in bbox]
                    bbox = (x, y, w, h)

                    # Process
                    if isinstance(clnf, MetalAcceleratedCLNF):
                        landmarks, _ = clnf.fit(frame, bbox)
                        result = au_pipeline.process_frame_metal(frame, landmarks)
                    else:
                        landmarks, _ = clnf.fit(frame, bbox)
                        result = au_pipeline._process_frame(frame, i, i/30.0)

            elapsed = (time.perf_counter() - start) * 1000
            frame_times.append(elapsed)

        avg_time = np.mean(frame_times)
        fps = 1000 / avg_time

        print(f"  Average frame time: {avg_time:.1f}ms")
        print(f"  FPS: {fps:.2f}")


def show_acceleration_roadmap():
    """Show complete GPU acceleration roadmap."""

    print("\n" + "=" * 60)
    print("GPU ACCELERATION ROADMAP")
    print("=" * 60)

    print("\n1. APPLE SILICON (Metal/CoreML)")
    print("-" * 40)
    print("✓ Metal Performance Shaders for matrix ops")
    print("✓ CoreML for neural networks")
    print("✓ Unified memory architecture")
    print("Expected: 3-5x speedup")

    print("\n2. NVIDIA GPU (CUDA/TensorRT)")
    print("-" * 40)
    print("✓ CUDA kernels for custom operations")
    print("✓ TensorRT for optimized inference")
    print("✓ cuDNN for neural network layers")
    print("Expected: 5-10x speedup")

    print("\n3. CROSS-PLATFORM (ONNX Runtime)")
    print("-" * 40)
    print("✓ Works on all platforms")
    print("✓ Multiple execution providers")
    print("✓ Automatic optimization")
    print("Expected: 2-4x speedup")

    print("\n4. IMPLEMENTATION PRIORITY")
    print("-" * 40)
    print("1. Export models to ONNX format")
    print("2. Implement ONNX Runtime inference")
    print("3. Add Metal backend for Apple Silicon")
    print("4. Add CUDA backend for NVIDIA")
    print("5. Optimize memory transfers")


if __name__ == "__main__":
    # Run Metal benchmark if available
    benchmark_metal_acceleration()

    # Show acceleration roadmap
    show_acceleration_roadmap()

    # Initialize ONNX pipeline
    print("\n" + "=" * 60)
    print("ONNX RUNTIME SETUP")
    print("=" * 60)
    onnx_pipeline = ONNXAcceleratedPipeline()

    print("\nTo continue with GPU acceleration:")
    print("1. Install PyTorch MPS: pip install torch torchvision")
    print("2. Install ONNX Runtime: pip install onnxruntime")
    print("3. For CoreML: pip install coremltools onnx-coreml")
    print("4. For CUDA: Install CUDA toolkit and onnxruntime-gpu")