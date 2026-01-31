#!/usr/bin/env python3
"""
WFLW 98-point to Dlib 68-point Landmark Mapper

Maps Dynaface/SPIGA 98-point WFLW landmarks to dlib 68-point format
for compatibility with pyfaceau AU prediction pipeline.

WFLW landmark layout (98 points):
- 0-32: Jawline/contour (33 points)
- 33-41: Right eyebrow (9 points)
- 42-50: Left eyebrow (9 points)
- 51-59: Nose (9 points)
- 60-67: Right eye (8 points)
- 68-75: Left eye (8 points)
- 76-87: Outer mouth (12 points)
- 88-95: Inner mouth (8 points)
- 96-97: Pupils (2 points) - not used in dlib

Dlib landmark layout (68 points):
- 0-16: Jawline (17 points)
- 17-21: Right eyebrow (5 points)
- 22-26: Left eyebrow (5 points)
- 27-35: Nose (9 points)
- 36-41: Right eye (6 points)
- 42-47: Left eye (6 points)
- 48-59: Outer mouth (12 points)
- 60-67: Inner mouth (8 points)
"""

import numpy as np
from typing import Optional


# WFLW index -> Dlib index mapping
# This maps specific WFLW landmark indices to their dlib equivalents
WFLW_TO_DLIB_MAP = {
    # Jawline: WFLW 0-32 (33 pts) → Dlib 0-16 (17 pts)
    # Sample every ~2 points to reduce from 33 to 17
    0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7, 16: 8,
    18: 9, 20: 10, 22: 11, 24: 12, 26: 13, 28: 14, 30: 15, 32: 16,

    # WFLW 33-37 (image left = subject's RIGHT eyebrow) → Dlib 22-26 (subject's RIGHT)
    33: 22, 34: 23, 35: 24, 36: 25, 37: 26,

    # WFLW 42-46 (image right = subject's LEFT eyebrow) → Dlib 17-21 (subject's LEFT)
    42: 17, 43: 18, 44: 19, 45: 20, 46: 21,

    # Nose: WFLW 51-59 (9 pts) → Dlib 27-35 (9 pts)
    # Direct 1:1 mapping
    51: 27, 52: 28, 53: 29, 54: 30, 55: 31, 56: 32, 57: 33, 58: 34, 59: 35,

    # Right eye: WFLW 60-67 (8 pts) → Dlib 36-41 (6 pts)
    # Skip points 62 and 66 (top/bottom mid-points)
    60: 36, 61: 37, 63: 38, 64: 39, 65: 40, 67: 41,

    # Left eye: WFLW 68-75 (8 pts) → Dlib 42-47 (6 pts)
    # Skip points 70 and 74 (top/bottom mid-points)
    68: 42, 69: 43, 71: 44, 72: 45, 73: 46, 75: 47,

    # Mouth outer: WFLW 76-87 (12 pts) → Dlib 48-59 (12 pts)
    # Direct 1:1 mapping
    76: 48, 77: 49, 78: 50, 79: 51, 80: 52, 81: 53,
    82: 54, 83: 55, 84: 56, 85: 57, 86: 58, 87: 59,

    # Mouth inner: WFLW 88-95 (8 pts) → Dlib 60-67 (8 pts)
    # Direct 1:1 mapping
    88: 60, 89: 61, 90: 62, 91: 63, 92: 64, 93: 65, 94: 66, 95: 67,
}

# Create reverse mapping for validation
DLIB_TO_WFLW_MAP = {v: k for k, v in WFLW_TO_DLIB_MAP.items()}


def wflw_to_dlib_68(wflw_landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert 98-point WFLW landmarks to 68-point dlib format.

    Args:
        wflw_landmarks: (98, 2) array of WFLW landmark coordinates

    Returns:
        (68, 2) array of dlib-format landmark coordinates, or None if invalid input
    """
    if wflw_landmarks is None:
        return None

    wflw_landmarks = np.asarray(wflw_landmarks)

    if wflw_landmarks.shape[0] < 96:  # Minimum required for mapping
        return None

    # Create output array
    dlib_landmarks = np.zeros((68, 2), dtype=np.float32)

    # Apply mapping
    for wflw_idx, dlib_idx in WFLW_TO_DLIB_MAP.items():
        if wflw_idx < wflw_landmarks.shape[0]:
            dlib_landmarks[dlib_idx] = wflw_landmarks[wflw_idx]

    return dlib_landmarks


def validate_mapping(wflw_landmarks: np.ndarray, dlib_landmarks: np.ndarray) -> dict:
    """
    Validate the landmark mapping by checking anatomical consistency.

    Args:
        wflw_landmarks: Original 98-point landmarks
        dlib_landmarks: Mapped 68-point landmarks

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'warnings': [],
        'stats': {}
    }

    if wflw_landmarks is None or dlib_landmarks is None:
        results['valid'] = False
        results['warnings'].append("Null landmarks provided")
        return results

    # Check that all 68 points are non-zero (were mapped)
    zero_points = np.sum(np.all(dlib_landmarks == 0, axis=1))
    if zero_points > 0:
        results['warnings'].append(f"{zero_points} points are at origin (unmapped)")

    # Calculate face bounding box from jaw landmarks
    jaw = dlib_landmarks[0:17]
    if not np.all(jaw == 0):
        face_width = np.max(jaw[:, 0]) - np.min(jaw[:, 0])
        face_height = np.max(jaw[:, 1]) - np.min(jaw[:, 1])
        results['stats']['face_width'] = float(face_width)
        results['stats']['face_height'] = float(face_height)

    # Check eye landmarks form closed shapes
    right_eye = dlib_landmarks[36:42]
    left_eye = dlib_landmarks[42:48]

    def check_closed_contour(points, name):
        """Check if points form a reasonable closed contour."""
        if np.all(points == 0):
            results['warnings'].append(f"{name} landmarks are all zero")
            return

        # Calculate perimeter
        perimeter = 0
        for i in range(len(points)):
            perimeter += np.linalg.norm(points[i] - points[(i + 1) % len(points)])

        # Calculate area using shoelace formula
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        results['stats'][f'{name}_perimeter'] = float(perimeter)
        results['stats'][f'{name}_area'] = float(area)

        # Check for degenerate shapes
        if area < 1.0:
            results['warnings'].append(f"{name} has very small area ({area:.2f})")

    check_closed_contour(right_eye, 'right_eye')
    check_closed_contour(left_eye, 'left_eye')

    # Check mouth landmarks
    outer_mouth = dlib_landmarks[48:60]
    inner_mouth = dlib_landmarks[60:68]

    check_closed_contour(outer_mouth, 'outer_mouth')
    check_closed_contour(inner_mouth, 'inner_mouth')

    # Overall validity
    if len(results['warnings']) > 2:
        results['valid'] = False

    return results


def get_mapping_info() -> dict:
    """
    Get information about the WFLW to dlib mapping.

    Returns:
        Dictionary with mapping statistics and details
    """
    return {
        'wflw_points': 98,
        'dlib_points': 68,
        'mapped_points': len(WFLW_TO_DLIB_MAP),
        'regions': {
            'jawline': {'wflw': '0-32', 'dlib': '0-16', 'subsample': True},
            'right_eyebrow': {'wflw': '33-41', 'dlib': '17-21', 'subsample': True},
            'left_eyebrow': {'wflw': '42-50', 'dlib': '22-26', 'subsample': True},
            'nose': {'wflw': '51-59', 'dlib': '27-35', 'subsample': False},
            'right_eye': {'wflw': '60-67', 'dlib': '36-41', 'subsample': True},
            'left_eye': {'wflw': '68-75', 'dlib': '42-47', 'subsample': True},
            'outer_mouth': {'wflw': '76-87', 'dlib': '48-59', 'subsample': False},
            'inner_mouth': {'wflw': '88-95', 'dlib': '60-67', 'subsample': False},
        }
    }


if __name__ == "__main__":
    # Test the mapping
    print("Testing WFLW to Dlib landmark mapping...")

    # Create dummy 98-point landmarks
    test_landmarks = np.random.rand(98, 2) * 100

    # Convert
    result = wflw_to_dlib_68(test_landmarks)

    print(f"Input shape: {test_landmarks.shape}")
    print(f"Output shape: {result.shape}")

    # Validate
    validation = validate_mapping(test_landmarks, result)
    print(f"Validation: {validation}")

    # Show mapping info
    info = get_mapping_info()
    print(f"\nMapping info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
