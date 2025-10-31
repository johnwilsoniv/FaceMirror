#!/usr/bin/env python3
"""
FHOG (Felzenszwalb HOG) Feature Extractor

Extracts FHOG features using dlib, matching OpenFace 2.2's implementation.

OpenFace 2.2 uses dlib's extract_fhog_features which produces 31-bin descriptors
per cell. The features are then flattened to a 1D vector.

Reference: OpenFace/lib/local/FaceAnalyser/src/Face_utils.cpp:238
"""

import numpy as np
import dlib
import cv2
from typing import Tuple


class FHOGExtractor:
    """
    Extracts FHOG features compatible with OpenFace 2.2 SVR models
    """

    def __init__(self, cell_size: int = 8):
        """
        Initialize FHOG extractor

        Args:
            cell_size: Size of each HOG cell in pixels (default: 8)
                      OpenFace typically uses 8
        """
        self.cell_size = cell_size

    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Extract FHOG features from an image

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            Tuple of:
            - descriptor: Flattened FHOG descriptor (1D array of shape (n_cells*31,))
            - num_rows: Number of vertical cells
            - num_cols: Number of horizontal cells
        """
        # Convert to format dlib can use
        if len(image.shape) == 2:
            # Grayscale
            dlib_img = dlib.array(image)
        elif image.shape[2] == 3:
            # OpenCV uses BGR, but dlib extract_fhog_features works with any 3-channel
            dlib_img = dlib.array(image)
        else:
            raise ValueError(f"Unsupported image format: shape={image.shape}")

        # Extract FHOG features using dlib
        # Returns: array2d<matrix<float,31,1>>
        # Each cell contains a 31-dimensional feature vector
        hog = dlib.get_frontal_face_detector()  # Just to import dlib properly

        # Use dlib's extract_fhog_features
        # Note: dlib.extract_fhog_features is not directly exposed in Python bindings
        # We need to use a different approach or C++ extension

        raise NotImplementedError(
            "dlib's extract_fhog_features is not exposed in Python bindings. "
            "Need to either:\n"
            "1. Create a C++ extension wrapping dlib::extract_fhog_features\n"
            "2. Use an alternative FHOG implementation (skimage, OpenCV)\n"
            "3. Use dlib's fhog_object_detector which includes FHOG internally"
        )

    def _flatten_hog(self, hog, num_rows: int, num_cols: int) -> np.ndarray:
        """
        Flatten HOG descriptors to 1D array matching OpenFace format

        OpenFace flattens in this order (Face_utils.cpp:259-265):
        for y in range(num_cols):
            for x in range(num_rows):
                for o in range(31):
                    descriptor[idx++] = hog[y][x](o)

        Args:
            hog: HOG descriptors from dlib
            num_rows: Number of vertical cells
            num_cols: Number of horizontal cells

        Returns:
            Flattened descriptor (num_cols * num_rows * 31,)
        """
        descriptor = np.zeros(num_cols * num_rows * 31, dtype=np.float64)
        idx = 0

        for y in range(num_cols):
            for x in range(num_rows):
                for o in range(31):
                    descriptor[idx] = hog[y][x][o]
                    idx += 1

        return descriptor


def test_dlib_availability():
    """Test if dlib FHOG extraction is available"""
    print("="*80)
    print("Testing dlib FHOG availability")
    print("="*80)

    # Check dlib version
    import dlib
    print(f"\ndlib version: {dlib.__version__}")

    # List available dlib functions
    print("\nSearching for FHOG-related functions in dlib:")
    fhog_attrs = [attr for attr in dir(dlib) if 'fhog' in attr.lower() or 'hog' in attr.lower()]

    if fhog_attrs:
        print(f"Found: {fhog_attrs}")
    else:
        print("No FHOG functions found in Python bindings")

    print("\n" + "="*80)
    print("Next steps:")
    print("="*80)
    print("1. Check if dlib.fhog_object_detector can be used")
    print("2. Consider using skimage.feature.hog as alternative")
    print("3. Or create C++ extension wrapping dlib::extract_fhog_features")


if __name__ == "__main__":
    test_dlib_availability()
