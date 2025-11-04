#!/usr/bin/env python3
"""
Compare three landmark detection pipelines:

Group 1 (Gold Standard): C++ OpenFace 2.2 (MTCNN + CLNF)
Group 2 (Current S1): PyFaceAU (RetinaFace + PFLD + CLNF refinement)
Group 3 (New): PyfaceLM (C++ wrapper) landmarks

This script compares landmark accuracy and investigates AU extraction compatibility.
"""

import sys
import numpy as np
import cv2
from pathlib import Path
import subprocess
import tempfile

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "PyfaceLM"))
sys.path.insert(0, str(Path(__file__).parent.parent / "archive_python_implementation/pyfaceau"))

from pyfacelm import CLNFDetector
from pyfaceau_detector import PyFaceAU68LandmarkDetector


class ThreePipelineComparison:
    """Compare three different landmark detection pipelines"""

    def __init__(self):
        """Initialize all three pipelines"""
        print("="*70)
        print("THREE-PIPELINE LANDMARK COMPARISON")
        print("="*70)

        # Pipeline 1: C++ OpenFace 2.2
        self.openface_binary = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction")
        if not self.openface_binary.exists():
            raise FileNotFoundError(f"OpenFace binary not found: {self.openface_binary}")
        print(f"\n✓ Pipeline 1: C++ OpenFace 2.2")
        print(f"  Binary: {self.openface_binary}")

        # Pipeline 2: Current S1 (PyFaceAU)
        print(f"\n✓ Pipeline 2: S1 Current (PyFaceAU)")
        self.s1_detector = PyFaceAU68LandmarkDetector(
            debug_mode=False,
            use_clnf_refinement=True
        )
        print(f"  Detector: PyFaceAU (RetinaFace + PFLD + CLNF)")

        # Pipeline 3: PyfaceLM
        print(f"\n✓ Pipeline 3: PyfaceLM (New)")
        self.pyfacelm_detector = CLNFDetector()
        print(f"  Detector: C++ OpenFace wrapper")

        print("\n" + "="*70)

    def detect_group1_cpp_openface(self, image_path):
        """
        Group 1: C++ OpenFace 2.2 full pipeline

        Returns:
            landmarks: (68, 2) array
            confidence: float
            metadata: dict with timing, etc.
        """
        import time

        print(f"\n  [Group 1] C++ OpenFace 2.2...")
        start = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            cmd = [
                str(self.openface_binary),
                "-f", str(image_path),
                "-out_dir", str(tmpdir),
                "-2Dfp",  # 2D landmarks only
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"    ✗ Failed: {result.stderr[:200]}")
                return None, None, {"error": "C++ binary failed"}

            # Parse CSV
            csv_file = tmpdir / f"{Path(image_path).stem}.csv"
            if not csv_file.exists():
                return None, None, {"error": "No CSV output"}

            landmarks, confidence = self._parse_openface_csv(csv_file)

        elapsed = time.time() - start

        print(f"    ✓ Success: {landmarks.shape} landmarks, conf={confidence:.3f}, time={elapsed:.3f}s")

        return landmarks, confidence, {
            "time": elapsed,
            "method": "C++ OpenFace 2.2 (MTCNN + CLNF)"
        }

    def detect_group2_s1_current(self, image_path):
        """
        Group 2: S1 Current pipeline (PyFaceAU)

        Returns:
            landmarks: (68, 2) array
            confidence: float
            metadata: dict
        """
        import time

        print(f"\n  [Group 2] S1 Current (PyFaceAU)...")
        start = time.time()

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None, {"error": "Failed to load image"}

        # Detect landmarks
        result = self.s1_detector.detect_landmarks(image)

        if result is None or 'landmarks' not in result:
            print(f"    ✗ Failed: No landmarks detected")
            return None, None, {"error": "Detection failed"}

        landmarks = result['landmarks']
        confidence = result.get('confidence', 0.0)

        elapsed = time.time() - start

        print(f"    ✓ Success: {landmarks.shape} landmarks, conf={confidence:.3f}, time={elapsed:.3f}s")

        return landmarks, confidence, {
            "time": elapsed,
            "method": "S1 PyFaceAU (RetinaFace + PFLD + CLNF)"
        }

    def detect_group3_pyfacelm(self, image_path):
        """
        Group 3: PyfaceLM wrapper

        Returns:
            landmarks: (68, 2) array
            confidence: float
            metadata: dict
        """
        import time

        print(f"\n  [Group 3] PyfaceLM...")
        start = time.time()

        # Detect landmarks
        landmarks, confidence, bbox = self.pyfacelm_detector.detect(image_path)

        elapsed = time.time() - start

        print(f"    ✓ Success: {landmarks.shape} landmarks, conf={confidence:.3f}, time={elapsed:.3f}s")

        return landmarks, confidence, {
            "time": elapsed,
            "method": "PyfaceLM (C++ OpenFace wrapper)",
            "bbox": bbox
        }

    def _parse_openface_csv(self, csv_file):
        """Parse OpenFace CSV to extract 68 landmarks and confidence"""
        with open(csv_file, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return None, None

        header = lines[0].strip().split(',')
        values = lines[1].strip().split(',')

        # Extract 68 landmarks
        landmarks = []
        for i in range(68):
            try:
                x_idx = header.index(f'x_{i}')
                y_idx = header.index(f'y_{i}')
            except ValueError:
                try:
                    x_idx = header.index(f' x_{i}')
                    y_idx = header.index(f' y_{i}')
                except ValueError:
                    return None, None

            x = float(values[x_idx])
            y = float(values[y_idx])
            landmarks.append([x, y])

        landmarks = np.array(landmarks, dtype=np.float32)

        # Extract confidence
        try:
            conf_idx = header.index('confidence')
        except ValueError:
            conf_idx = header.index(' confidence')

        confidence = float(values[conf_idx])

        return landmarks, confidence

    def compare_all(self, image_path):
        """
        Run all three pipelines and compare results

        Args:
            image_path: Path to test image

        Returns:
            dict with results from all three pipelines
        """
        image_path = Path(image_path)

        print(f"\n{'='*70}")
        print(f"Testing: {image_path.name}")
        print(f"{'='*70}")

        # Run all three pipelines
        lm1, conf1, meta1 = self.detect_group1_cpp_openface(image_path)
        lm2, conf2, meta2 = self.detect_group2_s1_current(image_path)
        lm3, conf3, meta3 = self.detect_group3_pyfacelm(image_path)

        # Calculate errors (using Group 1 as ground truth)
        print(f"\n{'='*70}")
        print("COMPARISON RESULTS")
        print(f"{'='*70}")

        results = {
            "image": str(image_path),
            "group1_cpp": {"landmarks": lm1, "confidence": conf1, "metadata": meta1},
            "group2_s1": {"landmarks": lm2, "confidence": conf2, "metadata": meta2},
            "group3_pyfacelm": {"landmarks": lm3, "confidence": conf3, "metadata": meta3}
        }

        # Compare Group 2 vs Group 1
        if lm1 is not None and lm2 is not None:
            error_2vs1 = np.abs(lm2 - lm1)
            mean_error = error_2vs1.mean()
            max_error = error_2vs1.max()
            rmse = np.sqrt((error_2vs1 ** 2).mean())

            print(f"\nGroup 2 (S1 Current) vs Group 1 (Gold Standard):")
            print(f"  Mean error: {mean_error:.3f} px")
            print(f"  Max error:  {max_error:.3f} px")
            print(f"  RMSE:       {rmse:.3f} px")

            results["error_2vs1"] = {
                "mean": float(mean_error),
                "max": float(max_error),
                "rmse": float(rmse)
            }

        # Compare Group 3 vs Group 1
        if lm1 is not None and lm3 is not None:
            error_3vs1 = np.abs(lm3 - lm1)
            mean_error = error_3vs1.mean()
            max_error = error_3vs1.max()
            rmse = np.sqrt((error_3vs1 ** 2).mean())

            print(f"\nGroup 3 (PyfaceLM) vs Group 1 (Gold Standard):")
            print(f"  Mean error: {mean_error:.3f} px")
            print(f"  Max error:  {max_error:.3f} px")
            print(f"  RMSE:       {rmse:.3f} px")

            results["error_3vs1"] = {
                "mean": float(mean_error),
                "max": float(max_error),
                "rmse": float(rmse)
            }

        # Performance comparison
        print(f"\nPerformance:")
        if meta1: print(f"  Group 1 (C++ OpenFace):  {meta1.get('time', 0):.3f}s")
        if meta2: print(f"  Group 2 (S1 Current):    {meta2.get('time', 0):.3f}s")
        if meta3: print(f"  Group 3 (PyfaceLM):      {meta3.get('time', 0):.3f}s")

        return results

    def visualize_comparison(self, image_path, results, output_path):
        """Create side-by-side visualization of all three pipelines"""
        import cv2

        # Load image
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        # Create three copies
        img1 = img.copy()
        img2 = img.copy()
        img3 = img.copy()

        # Draw landmarks for each group
        lm1 = results["group1_cpp"]["landmarks"]
        lm2 = results["group2_s1"]["landmarks"]
        lm3 = results["group3_pyfacelm"]["landmarks"]

        if lm1 is not None:
            for x, y in lm1:
                cv2.circle(img1, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(img1, "Group 1: C++ OpenFace", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(img1, f"Conf: {results['group1_cpp']['confidence']:.3f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if lm2 is not None:
            for x, y in lm2:
                cv2.circle(img2, (int(x), int(y)), 2, (255, 0, 0), -1)
            cv2.putText(img2, "Group 2: S1 Current", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv2.putText(img2, f"Conf: {results['group2_s1']['confidence']:.3f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            if "error_2vs1" in results:
                cv2.putText(img2, f"Error: {results['error_2vs1']['mean']:.2f}px", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if lm3 is not None:
            for x, y in lm3:
                cv2.circle(img3, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.putText(img3, "Group 3: PyfaceLM", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(img3, f"Conf: {results['group3_pyfacelm']['confidence']:.3f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if "error_3vs1" in results:
                cv2.putText(img3, f"Error: {results['error_3vs1']['mean']:.2f}px", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Combine side by side
        comparison = np.hstack([img1, img2, img3])

        # Save
        cv2.imwrite(str(output_path), comparison)
        print(f"\n✓ Visualization saved: {output_path}")


def main():
    """Run three-pipeline comparison on test images"""

    # Initialize comparison
    comparison = ThreePipelineComparison()

    # Test images
    test_images = [
        "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg",
        "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_9330.jpg",
    ]

    # Output directory
    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/test_output")
    output_dir.mkdir(exist_ok=True)

    # Run comparison on each image
    all_results = []
    for image_path in test_images:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"✗ Skipping {image_path.name} (not found)")
            continue

        # Run comparison
        results = comparison.compare_all(image_path)
        all_results.append(results)

        # Create visualization
        output_path = output_dir / f"{image_path.stem}_three_pipeline_comparison.jpg"
        comparison.visualize_comparison(image_path, results, output_path)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for results in all_results:
        print(f"\n{Path(results['image']).name}:")
        if "error_2vs1" in results:
            print(f"  Group 2 (S1) error:     {results['error_2vs1']['mean']:.3f}px")
        if "error_3vs1" in results:
            print(f"  Group 3 (PyfaceLM) error: {results['error_3vs1']['mean']:.3f}px")

    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
