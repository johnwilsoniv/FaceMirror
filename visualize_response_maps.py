#!/usr/bin/env python3
"""
Response Map Visualization Tool

Extracts and visualizes CLNF patch expert response maps to debug convergence issues.
This script instruments the optimizer to capture response maps at each iteration
and generates heatmap visualizations overlaid on the face image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.insert(0, 'pyclnf')
from pyclnf import CLNF
from pyclnf.core.optimizer import NURLMSOptimizer

# Test configuration
VIDEO_PATH = 'Patient Data/Normal Cohort/IMG_0433.MOV'
FRAME_NUM = 50
FACE_BBOX = (241, 555, 532, 532)  # (x, y, w, h)
OUTPUT_DIR = Path('test_output/response_maps')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class InstrumentedOptimizer(NURLMSOptimizer):
    """
    Instrumented optimizer that captures response maps during fitting.

    This subclass intercepts _compute_response_map calls and saves the
    response maps for visualization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_maps_history = []  # List of dicts per iteration
        self.current_iteration = 0

    def optimize(self, *args, **kwargs):
        """Wrap optimize to track iterations."""
        self.response_maps_history = []
        self.current_iteration = 0
        return super().optimize(*args, **kwargs)

    def _compute_mean_shift(self, *args, **kwargs):
        """Intercept mean-shift computation to capture response maps."""
        # Extract arguments
        landmarks_2d = args[0]
        patch_experts = args[1]
        image = args[2]
        pdm = args[3]
        window_size = args[4]
        sim_img_to_ref = args[5] if len(args) > 5 else kwargs.get('sim_img_to_ref')
        sim_ref_to_img = args[6] if len(args) > 6 else kwargs.get('sim_ref_to_img')
        sigma_components = args[7] if len(args) > 7 else kwargs.get('sigma_components')

        # Storage for this iteration
        iteration_data = {
            'iteration': self.current_iteration,
            'window_size': window_size,
            'landmarks': landmarks_2d.copy(),
            'response_maps': {},
            'peak_locations': {},
            'peak_values': {}
        }

        n_points = landmarks_2d.shape[0]

        # For each landmark, compute and save response map
        for landmark_idx, patch_expert in patch_experts.items():
            if landmark_idx >= n_points:
                continue

            lm_x, lm_y = landmarks_2d[landmark_idx]

            # Compute response map
            response_map = self._compute_response_map(
                image, lm_x, lm_y, patch_expert, window_size,
                sim_img_to_ref, sim_ref_to_img, sigma_components
            )

            if response_map is not None:
                # Store response map
                iteration_data['response_maps'][landmark_idx] = response_map.copy()

                # Find peak location
                peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
                peak_y, peak_x = peak_idx  # row, col = y, x
                peak_value = response_map[peak_y, peak_x]

                # Compute offset from center
                resp_size = response_map.shape[0]
                center = (resp_size - 1) / 2.0
                offset_x = peak_x - center
                offset_y = peak_y - center

                iteration_data['peak_locations'][landmark_idx] = (offset_x, offset_y)
                iteration_data['peak_values'][landmark_idx] = peak_value

        # Save iteration data
        self.response_maps_history.append(iteration_data)
        self.current_iteration += 1

        # Call original implementation
        return super()._compute_mean_shift(
            landmarks_2d, patch_experts, image, pdm, window_size,
            sim_img_to_ref, sim_ref_to_img, sigma_components
        )


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def visualize_response_map_grid(frame, iteration_data, output_path, max_landmarks=16):
    """
    Visualize response maps as a grid of heatmaps.

    Shows response maps for the landmarks with largest peak offsets.
    """
    response_maps = iteration_data['response_maps']
    peak_locations = iteration_data['peak_locations']
    landmarks = iteration_data['landmarks']

    if len(response_maps) == 0:
        return

    # Sort landmarks by peak offset magnitude (worst first)
    landmark_indices = sorted(
        response_maps.keys(),
        key=lambda idx: np.sqrt(peak_locations[idx][0]**2 + peak_locations[idx][1]**2),
        reverse=True
    )

    # Take worst N landmarks
    landmark_indices = landmark_indices[:max_landmarks]
    n_landmarks = len(landmark_indices)

    # Create grid
    n_cols = 4
    n_rows = (n_landmarks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, landmark_idx in enumerate(landmark_indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        response_map = response_maps[landmark_idx]
        offset_x, offset_y = peak_locations[landmark_idx]
        lm_x, lm_y = landmarks[landmark_idx]

        # Plot heatmap
        im = ax.imshow(response_map, cmap='hot', interpolation='nearest')

        # Mark center (current landmark position)
        center = (response_map.shape[1] - 1) / 2.0
        ax.plot(center, center, 'b+', markersize=10, markeredgewidth=2, label='Current pos')

        # Mark peak
        peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
        ax.plot(peak_idx[1], peak_idx[0], 'gx', markersize=10, markeredgewidth=2, label='Peak')

        # Title with offset info
        ax.set_title(
            f'LM{landmark_idx} @ ({lm_x:.0f},{lm_y:.0f})\n'
            f'Offset: ({offset_x:+.1f}, {offset_y:+.1f})px',
            fontsize=10
        )
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for i in range(n_landmarks, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.suptitle(
        f'Response Maps - Iteration {iteration_data["iteration"]} (WS={iteration_data["window_size"]})\n'
        f'Showing {n_landmarks} landmarks with largest peak offsets',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_landmarks_with_offsets(frame, iteration_data, output_path):
    """
    Visualize landmarks overlaid on frame with peak offset vectors.
    """
    landmarks = iteration_data['landmarks']
    peak_locations = iteration_data['peak_locations']

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)

    # Draw landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='cyan', s=30, alpha=0.8, label='Landmarks')

    # Draw peak offset vectors for landmarks with significant offsets
    for landmark_idx, (offset_x, offset_y) in peak_locations.items():
        lm_x, lm_y = landmarks[landmark_idx]
        offset_mag = np.sqrt(offset_x**2 + offset_y**2)

        if offset_mag > 1.0:  # Only show significant offsets
            # Arrow from landmark to where peak thinks it should be
            ax.arrow(
                lm_x, lm_y, offset_x, offset_y,
                head_width=3, head_length=3,
                fc='red', ec='red', alpha=0.7, width=0.5
            )

            # Label worst offsets
            if offset_mag > 3.0:
                ax.text(
                    lm_x + offset_x, lm_y + offset_y,
                    f'{landmark_idx}\n({offset_mag:.1f}px)',
                    fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7)
                )

    ax.set_title(
        f'Landmarks with Response Map Peak Offsets\n'
        f'Iteration {iteration_data["iteration"]} (WS={iteration_data["window_size"]})\n'
        f'Red arrows show direction of peak offset',
        fontsize=14
    )
    ax.axis('off')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_statistics(iteration_data):
    """Print statistics about response maps."""
    peak_locations = iteration_data['peak_locations']
    peak_values = iteration_data['peak_values']

    if len(peak_locations) == 0:
        return

    # Compute offset magnitudes
    offsets = [(idx, np.sqrt(ox**2 + oy**2), ox, oy)
               for idx, (ox, oy) in peak_locations.items()]
    offsets.sort(key=lambda x: x[1], reverse=True)

    offset_mags = [x[1] for x in offsets]
    peak_vals = list(peak_values.values())

    print(f"\nIteration {iteration_data['iteration']} (WS={iteration_data['window_size']}):")
    print(f"  Peak offsets: mean={np.mean(offset_mags):.2f}px, "
          f"median={np.median(offset_mags):.2f}px, "
          f"max={np.max(offset_mags):.2f}px")
    print(f"  Peak values:  mean={np.mean(peak_vals):.6f}, "
          f"max={np.max(peak_vals):.6f}")

    # Show worst 5 offsets
    print(f"  Worst 5 peak offsets:")
    for i in range(min(5, len(offsets))):
        idx, mag, ox, oy = offsets[i]
        print(f"    LM{idx:2d}: offset=({ox:+5.2f}, {oy:+5.2f}) mag={mag:.2f}px")


def main():
    print("="*80)
    print("RESPONSE MAP VISUALIZATION")
    print("="*80)
    print()

    # Load frame
    print(f"Loading frame {FRAME_NUM} from {VIDEO_PATH}...")
    frame = extract_frame(VIDEO_PATH, FRAME_NUM)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Frame shape: {frame.shape}")
    print()

    # Create instrumented CLNF
    print("Initializing instrumented CLNF...")
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=5, window_sizes=[11])

    # Replace optimizer with instrumented version
    clnf.optimizer = InstrumentedOptimizer(
        regularization=clnf.optimizer.regularization,
        max_iterations=clnf.optimizer.max_iterations,
        convergence_threshold=clnf.optimizer.convergence_threshold,
        sigma=clnf.optimizer.sigma,
        weight_multiplier=clnf.optimizer.weight_multiplier
    )

    print("Running CLNF with response map capture...")
    landmarks, info = clnf.fit(gray, FACE_BBOX, return_params=True)
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final update: {info['final_update']:.6f}")
    print()

    # Extract captured response maps
    history = clnf.optimizer.response_maps_history
    print(f"Captured {len(history)} iterations of response maps")
    print()

    # Print statistics for each iteration
    for iteration_data in history:
        print_statistics(iteration_data)

    # Visualize each iteration
    print(f"\nGenerating visualizations...")
    for iteration_data in history:
        iter_num = iteration_data['iteration']
        ws = iteration_data['window_size']

        # Response map heatmap grid
        grid_path = OUTPUT_DIR / f'response_maps_iter{iter_num}_ws{ws}.png'
        visualize_response_map_grid(frame, iteration_data, grid_path)
        print(f"  Saved: {grid_path}")

        # Landmarks with offset vectors
        offset_path = OUTPUT_DIR / f'peak_offsets_iter{iter_num}_ws{ws}.png'
        visualize_landmarks_with_offsets(frame, iteration_data, offset_path)
        print(f"  Saved: {offset_path}")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print()
    print("Key findings to look for:")
    print("  1. Are response map peaks consistently offset in one direction?")
    print("  2. Do peak offsets decrease across iterations (should → 0)?")
    print("  3. Are certain face regions (eyes, mouth, jaw) worse than others?")
    print("  4. Do response maps have smooth single peaks or multiple peaks?")
    print()


if __name__ == "__main__":
    main()
