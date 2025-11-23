#!/usr/bin/env python3
"""
Test if flipping the row index when storing response map fixes the issue.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING

def compute_mean_shift(response_map, sigma=1.0):
    ws = response_map.shape[0]
    center = (ws - 1) / 2.0
    a_kde = -0.5 / (sigma * sigma)

    total_weight = mx = my = 0.0

    for ii in range(ws):
        for jj in range(ws):
            dist_sq = (ii - center)**2 + (jj - center)**2
            kde_weight = np.exp(a_kde * dist_sq)
            weight = response_map[ii, jj] * kde_weight
            total_weight += weight
            mx += weight * jj
            my += weight * ii

    if total_weight > 1e-10:
        return (mx / total_weight) - center, (my / total_weight) - center
    return 0.0, 0.0

def main():
    print("=" * 70)
    print("TEST RESPONSE MAP INDEX FLIP")
    print("=" * 70)

    # Load video frame
    video = cv2.VideoCapture('Patient Data/Normal Cohort/Shorty.mov')
    ret, frame = video.read()
    video.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load eye model
    model_dir = 'pyclnf/models'
    eye_model = HierarchicalEyeModel(model_dir)
    pdm = eye_model.pdm['left']

    # C++ input landmarks
    CPP_LEFT_EYE_INPUT = {
        36: (392.1590, 847.6613), 37: (410.0039, 828.3166),
        38: (436.9223, 826.1841), 39: (461.9583, 842.8420),
        40: (438.4380, 850.4288), 41: (411.4089, 853.9998)
    }

    target_points = np.array([CPP_LEFT_EYE_INPUT[i] for i in [36, 37, 38, 39, 40, 41]])
    params = eye_model._fit_eye_shape(target_points, LEFT_EYE_MAPPING, 'left', main_rotation=None)
    eye_landmarks = pdm.params_to_landmarks_2d(params)

    ccnf = eye_model.ccnf['left']
    patch_experts = ccnf.get_all_patch_experts(eye_model.patch_scale)

    ws = eye_model.window_sizes[0]
    half_ws = ws // 2

    CPP_DELTA = {
        36: (+0.4725, +2.5932), 37: (+0.4878, +2.8680),
        38: (+0.2048, +2.5201), 39: (-0.5040, +1.1696),
        40: (-0.0311, +1.9948), 41: (+0.4704, +2.0950)
    }

    eye_to_main = {8: 36, 10: 37, 12: 38, 14: 39, 16: 40, 18: 41}

    print("\nTesting different storage conventions:\n")
    print("Original: response_map[dy + half_ws, dx + half_ws] = response")
    print("Flip Y:   response_map[-dy + half_ws, dx + half_ws] = response")
    print("Flip X:   response_map[dy + half_ws, -dx + half_ws] = response")
    print("Flip both: response_map[-dy + half_ws, -dx + half_ws] = response")

    for eye_idx in [8, 10, 12, 14, 16, 18]:
        if eye_idx not in patch_experts:
            continue

        main_idx = eye_to_main[eye_idx]
        cpp_dy = CPP_DELTA[main_idx][1]
        lm = eye_landmarks[eye_idx]
        x_int, y_int = int(round(lm[0])), int(round(lm[1]))
        pe = patch_experts[eye_idx]

        # Compute response maps with different conventions
        orig_map = np.zeros((ws, ws), dtype=np.float32)
        flip_y_map = np.zeros((ws, ws), dtype=np.float32)
        flip_x_map = np.zeros((ws, ws), dtype=np.float32)
        flip_both_map = np.zeros((ws, ws), dtype=np.float32)

        for dy in range(-half_ws, half_ws + 1):
            for dx in range(-half_ws, half_ws + 1):
                px, py = x_int + dx, y_int + dy
                patch_size = pe.height
                half_patch = patch_size // 2

                y1 = max(0, py - half_patch)
                y2 = min(gray.shape[0], py + half_patch + 1)
                x1 = max(0, px - half_patch)
                x2 = min(gray.shape[1], px + half_patch + 1)

                if y2 - y1 < patch_size or x2 - x1 < patch_size:
                    patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
                    patch[:y2-y1, :x2-x1] = gray[y1:y2, x1:x2]
                else:
                    patch = gray[y1:y2, x1:x2]

                response = pe.compute_response(patch)

                # Different storage conventions
                orig_map[dy + half_ws, dx + half_ws] = response
                flip_y_map[-dy + half_ws, dx + half_ws] = response
                flip_x_map[dy + half_ws, -dx + half_ws] = response
                flip_both_map[-dy + half_ws, -dx + half_ws] = response

        # Compute mean-shifts
        orig_ms_x, orig_ms_y = compute_mean_shift(orig_map)
        flip_y_ms_x, flip_y_ms_y = compute_mean_shift(flip_y_map)
        flip_x_ms_x, flip_x_ms_y = compute_mean_shift(flip_x_map)
        flip_both_ms_x, flip_both_ms_y = compute_mean_shift(flip_both_map)

        orig_match = (orig_ms_y > 0) == (cpp_dy > 0) if abs(cpp_dy) > 0.1 else True
        flip_y_match = (flip_y_ms_y > 0) == (cpp_dy > 0) if abs(cpp_dy) > 0.1 else True
        flip_x_match = (flip_x_ms_x > 0) == (CPP_DELTA[main_idx][0] > 0) if abs(CPP_DELTA[main_idx][0]) > 0.1 else True
        flip_both_match = ((flip_both_ms_y > 0) == (cpp_dy > 0) if abs(cpp_dy) > 0.1 else True) and \
                         ((flip_both_ms_x > 0) == (CPP_DELTA[main_idx][0] > 0) if abs(CPP_DELTA[main_idx][0]) > 0.1 else True)

        print(f"\nEye_{eye_idx} (C++ Δy: {cpp_dy:+.4f}):")
        print(f"  Original:   ms_y={orig_ms_y:+.4f} {'✓' if orig_match else '✗'}")
        print(f"  Flip Y:     ms_y={flip_y_ms_y:+.4f} {'✓' if flip_y_match else '✗'}")

        if flip_y_match and not orig_match:
            print(f"  *** FLIP Y INDEX FIXES THIS! ***")

if __name__ == '__main__':
    main()
