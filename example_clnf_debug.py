#!/usr/bin/env python3
"""
Example script demonstrating CLNF debug mode usage.

This shows how to use the debug mode to get detailed information about
the CLNF optimization process, similar to MTCNN's detect_with_debug.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from pyclnf import CLNF

def print_debug_summary(debug_info):
    """Print a human-readable summary of debug information."""
    print("\n" + "="*80)
    print("CLNF DEBUG SUMMARY")
    print("="*80)

    # Initialization
    print("\n[INITIALIZATION]")
    print(f"  BBox: {debug_info['bbox']}")
    print(f"  Tracked landmarks: {debug_info['tracked_landmarks']}")

    init = debug_info['initialization']
    print(f"  Initial params shape: {init['params'].shape}")
    print(f"  Tracked landmark positions:")
    for idx, pos in init['tracked_landmarks'].items():
        print(f"    Landmark {idx}: ({pos[0]:.2f}, {pos[1]:.2f})")

    # Window stages
    print(f"\n[OPTIMIZATION] - {len(debug_info['window_stages'])} window stages")
    for stage in debug_info['window_stages']:
        ws = stage['window_size']
        scale = stage['patch_scale']
        opt = stage['optimization']

        print(f"\n  Window Size {ws} (scale={scale}):")
        print(f"    Iterations: {len(opt['iterations'])}")

        # Show first and last iteration for each tracked landmark
        if opt['iterations']:
            first_iter = opt['iterations'][0]
            last_iter = opt['iterations'][-1]

            print(f"    Iteration 0:")
            for lm_idx in debug_info['tracked_landmarks']:
                if lm_idx in first_iter.get('response_maps', {}):
                    resp = first_iter['response_maps'][lm_idx]
                    print(f"      LM{lm_idx}: pos=({resp['position'][0]:.2f}, {resp['position'][1]:.2f}), "
                          f"peak=({resp['response_peak']['row']}, {resp['response_peak']['col']}), "
                          f"peak_val={resp['response_peak']['value']:.6f}")

            if len(opt['iterations']) > 1:
                print(f"    Iteration {len(opt['iterations'])-1}:")
                for lm_idx in debug_info['tracked_landmarks']:
                    if lm_idx in last_iter.get('response_maps', {}):
                        resp = last_iter['response_maps'][lm_idx]
                        print(f"      LM{lm_idx}: pos=({resp['position'][0]:.2f}, {resp['position'][1]:.2f}), "
                              f"peak=({resp['response_peak']['row']}, {resp['response_peak']['col']}), "
                              f"peak_val={resp['response_peak']['value']:.6f}")

    # Final results
    print("\n[FINAL RESULTS]")
    final = debug_info['final']
    print(f"  Total iterations: {final['total_iterations']}")
    print(f"  Final tracked landmark positions:")
    for idx, pos in final['tracked_landmarks'].items():
        print(f"    Landmark {idx}: ({pos[0]:.2f}, {pos[1]:.2f})")

    print("="*80 + "\n")


def main():
    """Run CLNF with debug mode on a test image."""

    # Paths
    image_path = Path("calibration_frames/patient1_frame1.jpg")
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)

    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Initialize CLNF without detector (we'll provide bbox manually)
    print("Initializing CLNF model...")
    clnf = CLNF(detector=None)

    # Get bbox from known result (or you can use a detector)
    # For this example, we'll use a known good bbox
    bbox = (296, 778, 405, 407)  # x, y, w, h

    print(f"Using bbox: {bbox}")
    print("\nRunning CLNF with debug mode...")

    # Run CLNF with debug mode
    landmarks, info, debug_info = clnf.fit_with_debug(
        img,
        bbox,
        tracked_landmarks=[36, 48, 30, 8]  # left eye, mouth, nose, jaw
    )

    # Print summary
    print_debug_summary(debug_info)

    # Save full debug info as JSON
    debug_json_path = output_dir / "clnf_debug_info.json"
    print(f"Saving detailed debug info to: {debug_json_path}")

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    debug_json = convert_numpy(debug_info)
    with open(debug_json_path, 'w') as f:
        json.dump(debug_json, f, indent=2)

    # Save visualization
    vis_img = img.copy()
    for lm in landmarks:
        cv2.circle(vis_img, (int(lm[0]), int(lm[1])), 2, (0, 255, 0), -1)

    # Highlight tracked landmarks
    tracked = [36, 48, 30, 8]
    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255)]
    for idx, color in zip(tracked, colors):
        lm = landmarks[idx]
        cv2.circle(vis_img, (int(lm[0]), int(lm[1])), 5, color, -1)
        cv2.putText(vis_img, str(idx), (int(lm[0]) + 10, int(lm[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    vis_path = output_dir / "clnf_debug_visualization.jpg"
    cv2.imwrite(str(vis_path), vis_img)
    print(f"Saved visualization to: {vis_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
