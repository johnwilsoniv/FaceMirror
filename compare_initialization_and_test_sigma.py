#!/usr/bin/env python3
"""
Compare PyCLNF vs OpenFace C++ initialization and test sigma values.

Findings from OpenFace C++ source (LandmarkDetectorModel.cpp):
- Base sigma = 1.75 for main model (line 548)
- Dynamic adjustment: sigma += 0.25 * log(scale/0.25)/log(2) (line 778)
- For scale=0.25: sigma = 1.75 (no adjustment)
- For scale=0.5:  sigma = 1.75 + 0.25*1 = 2.0
- For scale=1.0:  sigma = 1.75 + 0.25*2 = 2.25
"""

import cv2
import numpy as np
import sys
import pandas as pd

sys.path.insert(0, 'pyclnf')
from pyclnf import CLNF

# Test configuration
VIDEO_PATH = 'Patient Data/Normal Cohort/IMG_0433.MOV'
FRAME_NUM = 50
FACE_BBOX = (241, 555, 532, 532)


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def load_openface_landmarks(csv_path='/tmp/openface_test/IMG_0433.csv'):
    """Load OpenFace C++ landmarks from CSV (frame 50)."""
    try:
        df = pd.read_csv(csv_path)
        # Get row for frame 50
        row = df[df['frame'] == FRAME_NUM].iloc[0]

        # Extract landmark columns
        x_cols = [c for c in df.columns if c.startswith('x_')]
        y_cols = [c for c in df.columns if c.startswith('y_')]

        landmarks = np.zeros((len(x_cols), 2))
        for i, (x_col, y_col) in enumerate(zip(sorted(x_cols), sorted(y_cols))):
            landmarks[i] = [row[x_col], row[y_col]]

        return landmarks
    except Exception as e:
        print(f"Could not load OpenFace landmarks: {e}")
        return None


def visualize_landmarks(frame, landmarks_dict, title="Landmark Comparison", output_path=None):
    """Visualize multiple landmark sets on the same image."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=14)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for idx, (name, landmarks) in enumerate(landmarks_dict.items()):
        if landmarks is not None:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax.scatter(landmarks[:, 0], landmarks[:, 1],
                      c=color, marker=marker, s=30, alpha=0.7, label=name)

    ax.legend(loc='upper right')
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    print("="*80)
    print("COMPARING INITIALIZATION AND TESTING SIGMA VALUES")
    print("="*80)
    print()

    # Load frame
    print(f"Loading frame {FRAME_NUM} from {VIDEO_PATH}...")
    frame = extract_frame(VIDEO_PATH, FRAME_NUM)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Frame shape: {frame.shape}")
    print()

    # ============================================================================
    # PART 1: COMPARE INITIALIZATIONS
    # ============================================================================
    print("="*80)
    print("PART 1: COMPARING INITIAL LANDMARK POSITIONS")
    print("="*80)
    print()

    # Get PyCLNF initial landmarks (0 iterations)
    print("Getting PyCLNF initial landmarks (0 iterations)...")
    clnf_init = CLNF(model_dir='pyclnf/models', max_iterations=0)
    pyclnf_init_landmarks, _ = clnf_init.fit(gray, FACE_BBOX, return_params=True)
    print(f"  PyCLNF initial landmarks: {pyclnf_init_landmarks.shape}")
    print()

    # Load OpenFace C++ final landmarks (after convergence)
    print("Loading OpenFace C++ final landmarks (fully converged)...")
    openface_final_landmarks = load_openface_landmarks()
    if openface_final_landmarks is not None:
        print(f"  OpenFace final landmarks: {openface_final_landmarks.shape}")
    print()

    # Compute initial error
    if openface_final_landmarks is not None:
        init_diff = pyclnf_init_landmarks - openface_final_landmarks
        init_diff_mag = np.linalg.norm(init_diff, axis=1)

        print("INITIAL LANDMARK ERROR (PyCLNF init vs OpenFace final):")
        print(f"  Mean error: {init_diff_mag.mean():.2f} px")
        print(f"  Median error: {np.median(init_diff_mag):.2f} px")
        print(f"  Max error: {init_diff_mag.max():.2f} px")
        print(f"  Min error: {init_diff_mag.min():.2f} px")
        print()

        # Find worst initialized landmarks
        worst_idx = np.argsort(-init_diff_mag)[:5]
        print("Worst 5 initialized landmarks:")
        for rank, idx in enumerate(worst_idx, 1):
            print(f"  {rank}. Landmark {idx:2d}: {init_diff_mag[idx]:.2f} px error")
        print()

    # ============================================================================
    # PART 2: TEST DIFFERENT SIGMA VALUES
    # ============================================================================
    print("="*80)
    print("PART 2: TESTING DIFFERENT SIGMA VALUES")
    print("="*80)
    print()

    print("OpenFace C++ sigma configuration:")
    print("  Base sigma = 1.75 (for 'inner' model)")
    print("  Dynamic adjustment: sigma += 0.25 * log(scale/0.25)/log(2)")
    print("  For scale=0.25: sigma = 1.75 (our test case)")
    print()

    # Test sigma values
    sigma_tests = [
        (1.5, "PyCLNF default (too small)"),
        (1.75, "OpenFace base sigma"),
        (2.0, "OpenFace @ scale=0.5"),
        (2.5, "Moderately larger"),
        (3.0, "Large (ws/3.7)"),
        (3.5, "OpenFace eyebrow sigma"),
        (4.0, "Very large (ws/2.75)"),
        (4.5, "Extremely large (ws/2.44)"),
    ]

    print(f"{'Sigma':<8} {'Description':<30} {'Converged':<12} {'Iters':<7} {'Final Update':<15} {'Ratio':<10}")
    print("-" * 95)

    results = []
    landmark_sets = {}

    for sigma, description in sigma_tests:
        # Test with this sigma (only window_size=11 for fair comparison)
        clnf = CLNF(model_dir='pyclnf/models', max_iterations=20, window_sizes=[11])
        clnf.optimizer.sigma = sigma

        landmarks, info = clnf.fit(gray, FACE_BBOX, return_params=True)

        # Display results
        ratio = info['final_update'] / 0.005
        converged_str = "YES" if info['converged'] else "NO"

        print(f"{sigma:<8.2f} {description:<30} {converged_str:<12} {info['iterations']:<7} {info['final_update']:<15.6f} {ratio:<10.1f}x")

        results.append({
            'sigma': sigma,
            'description': description,
            'converged': info['converged'],
            'iterations': info['iterations'],
            'final_update': info['final_update'],
            'ratio': ratio
        })

        # Save landmarks for visualization (only a few key values)
        if sigma in [1.5, 1.75, 3.0, 4.5]:
            landmark_sets[f"σ={sigma} ({description[:15]})"] = landmarks

    print()

    # ============================================================================
    # PART 3: ANALYSIS
    # ============================================================================
    print("="*80)
    print("PART 3: ANALYSIS")
    print("="*80)
    print()

    # Find best sigma
    best = min(results, key=lambda x: x['final_update'])
    print(f"BEST SIGMA: {best['sigma']}")
    print(f"  Description: {best['description']}")
    print(f"  Final update: {best['final_update']:.6f} (target: 0.005)")
    print(f"  Ratio: {best['ratio']:.1f}x the target")
    print(f"  Converged: {'YES' if best['converged'] else 'NO'}")
    print()

    # Compare with OpenFace sigma
    openface_sigma_result = next(r for r in results if r['sigma'] == 1.75)
    print(f"OPENFACE SIGMA (1.75) PERFORMANCE:")
    print(f"  Final update: {openface_sigma_result['final_update']:.6f}")
    print(f"  Ratio: {openface_sigma_result['ratio']:.1f}x the target")
    print(f"  Converged: {'YES' if openface_sigma_result['converged'] else 'NO'}")
    print()

    # Show improvement
    pyclnf_default = results[0]  # sigma=1.5
    improvement = (pyclnf_default['final_update'] - best['final_update']) / pyclnf_default['final_update'] * 100
    print(f"IMPROVEMENT OVER PYCLNF DEFAULT (σ=1.5):")
    print(f"  Before: {pyclnf_default['final_update']:.6f}")
    print(f"  After:  {best['final_update']:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print()

    # Check if converged with any sigma
    converged_sigmas = [r['sigma'] for r in results if r['converged']]
    if converged_sigmas:
        print(f"✓ CONVERGED with sigma: {converged_sigmas}")
    else:
        print(f"✗ DID NOT CONVERGE with any tested sigma value")
        print(f"  Closest: σ={best['sigma']} with final_update={best['final_update']:.6f}")
    print()

    # ============================================================================
    # PART 4: VISUALIZATIONS
    # ============================================================================
    print("="*80)
    print("PART 4: CREATING VISUALIZATIONS")
    print("="*80)
    print()

    # Add initial and final landmarks to visualization
    landmark_sets_full = {
        "PyCLNF init (0 iter)": pyclnf_init_landmarks,
    }
    if openface_final_landmarks is not None:
        landmark_sets_full["OpenFace final"] = openface_final_landmarks
    landmark_sets_full.update(landmark_sets)

    # Create visualization
    visualize_landmarks(
        frame,
        landmark_sets_full,
        title=f"Landmark Comparison: Initialization and Different Sigma Values\n(Frame {FRAME_NUM}, Bbox {FACE_BBOX})",
        output_path="initialization_and_sigma_comparison.png"
    )
    print()

    # ============================================================================
    # PART 5: RECOMMENDATIONS
    # ============================================================================
    print("="*80)
    print("PART 5: RECOMMENDATIONS")
    print("="*80)
    print()

    if best['sigma'] <= 2.0:
        print("RECOMMENDATION: Use OpenFace's sigma value (1.75)")
        print("  - Matches OpenFace C++ implementation")
        print("  - PyCLNF default (1.5) is slightly too small")
        print()
        print("Suggested fix in pyclnf/core/optimizer.py:")
        print("  Change: self.sigma = 1.5")
        print("  To:     self.sigma = 1.75  # Match OpenFace default")
    else:
        print(f"RECOMMENDATION: Use larger sigma ({best['sigma']})")
        print(f"  - {best['sigma']} performs {improvement:.1f}% better than default")
        print("  - Consider implementing OpenFace's dynamic sigma adjustment:")
        print()
        print("Suggested implementation:")
        print("```python")
        print("# In _compute_mean_shift method:")
        print("base_sigma = 1.75  # OpenFace default")
        print("scale_ratio = current_scale / 0.25")
        print("sigma = base_sigma + 0.25 * np.log(scale_ratio) / np.log(2)")
        print("a = -0.5 / (sigma * sigma)")
        print("```")

    print()
    print("NEXT STEPS:")
    print("1. Verify initialization matches OpenFace C++ (check visualization)")
    print("2. Implement dynamic sigma adjustment from OpenFace")
    print("3. Test with multiple scales to ensure consistent convergence")
    print()
    print("="*80)


if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("WARNING: matplotlib not available, skipping visualizations")

    main()
