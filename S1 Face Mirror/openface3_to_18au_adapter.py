#!/usr/bin/env python3
"""
OpenFace 3.0 to 18-AU Adapter

Converts OpenFace 3.0's 8 Action Unit output to the 18-AU structure
expected by downstream S2/S3 analysis pipeline, maintaining compatibility
with OpenFace 2.0 format.

OpenFace 3.0 provides:
- 8 AUs: AU1, AU2, AU4, AU6, AU12, AU15, AU20, AU25

S3 expects:
- 18 AUs: AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU14, AU15,
          AU16, AU17, AU20, AU23, AU25, AU26, AU45

Strategy:
- Direct mapping for available AUs
- NaN for unavailable AUs (including AU7)
- Calculate AU45 from eye landmarks using EAR
"""

import numpy as np
from au45_calculator import AU45Calculator


class OpenFace3To18AUAdapter:
    """
    Adapter to convert OpenFace 3.0's 8 AU output to 18-AU structure

    This maintains compatibility with the S2/S3 pipeline which expects
    the full 18-AU set from OpenFace 2.0.
    """

    def __init__(self):
        """Initialize the adapter with AU mapping configuration"""

        # OpenFace 3.0 AU mapping: tensor index -> AU name
        # Based on DISFA dataset subset (8 most common AUs)
        self.of3_au_mapping = {
            0: 'AU01_r',  # Inner Brow Raiser
            1: 'AU02_r',  # Outer Brow Raiser
            2: 'AU04_r',  # Brow Lowerer
            3: 'AU06_r',  # Cheek Raiser
            4: 'AU12_r',  # Lip Corner Puller (Smile)
            5: 'AU15_r',  # Lip Corner Depressor
            6: 'AU20_r',  # Lip Stretcher
            7: 'AU25_r',  # Lips Part
        }

        # Complete expected AU list in correct order (S3 requirement)
        self.expected_aus_r = [
            'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
            'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU16_r',
            'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
        ]

        # Binary (_c) versions of AUs (for compatibility, but all will be NaN)
        self.expected_aus_c = [
            'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c',
            'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU16_c',
            'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU45_c'
        ]

        # AU45 calculator for blink detection
        self.au45_calculator = AU45Calculator()

        # Track which AUs are available vs missing for logging
        self.available_aus = set(self.of3_au_mapping.values())
        self.missing_aus = set(self.expected_aus_r) - self.available_aus - {'AU45_r'}

    def adapt_frame(self, au_vector_8d, landmarks_98=None, frame_num=None):
        """
        Convert OpenFace 3.0's 8-AU output to 18-AU structure for one frame

        Args:
            au_vector_8d: (8,) numpy array or torch tensor from OpenFace 3.0
            landmarks_98: (98, 2) numpy array of WFLW landmarks (optional, for AU45)
            frame_num: Frame number for logging (optional)

        Returns:
            dict: Dictionary with all 18 AU intensity values (_r) and 18 binary values (_c)
                  Available AUs: float 0-5
                  Missing AUs: np.nan
                  AU45: calculated from landmarks or np.nan
        """
        # Convert torch tensor to numpy if needed
        if hasattr(au_vector_8d, 'detach'):
            au_vector_8d = au_vector_8d.detach().cpu().numpy()

        # Flatten if needed
        if au_vector_8d.ndim > 1:
            au_vector_8d = au_vector_8d.flatten()

        # Validate input
        if len(au_vector_8d) != 8:
            raise ValueError(f"Expected 8 AUs from OpenFace 3.0, got {len(au_vector_8d)}")

        # Initialize output dictionary
        adapted_aus = {}

        # Map available AUs from OpenFace 3.0
        for idx, au_name in self.of3_au_mapping.items():
            try:
                value = float(au_vector_8d[idx])
                # Clamp to valid range and handle any invalid values
                if np.isnan(value) or np.isinf(value):
                    adapted_aus[au_name] = np.nan
                else:
                    adapted_aus[au_name] = np.clip(value, 0.0, 5.0)
            except (IndexError, ValueError, TypeError):
                adapted_aus[au_name] = np.nan

        # Calculate AU45 from landmarks if available
        if landmarks_98 is not None:
            try:
                # Calculate AU45 without debug output for clean progress display
                au45_value = self.au45_calculator.calculate_au45_from_landmarks(landmarks_98, debug=False)
                adapted_aus['AU45_r'] = au45_value
            except Exception as e:
                # Silent failure - log only to avoid progress spam
                adapted_aus['AU45_r'] = np.nan
        else:
            adapted_aus['AU45_r'] = np.nan

        # Fill in missing AUs with NaN (including AU07)
        for au_name in self.expected_aus_r:
            if au_name not in adapted_aus:
                adapted_aus[au_name] = np.nan

        # Add binary versions (_c) - all NaN since OpenFace 3.0 doesn't provide these
        for au_name_c in self.expected_aus_c:
            adapted_aus[au_name_c] = np.nan

        # Add metadata for tracking
        adapted_aus['_au_source'] = 'OpenFace3.0'
        adapted_aus['_au07_status'] = 'missing'  # S3 will use AU06 instead

        return adapted_aus

    def get_csv_row_dict(self, au_vector_8d, landmarks_98, frame_num, timestamp,
                         confidence=1.0, success=1):
        """
        Create a complete CSV row dictionary for one frame

        This includes all metadata and AU values in the correct order for
        OpenFace 2.0-compatible CSV output.

        Args:
            au_vector_8d: (8,) array of AU values from OpenFace 3.0
            landmarks_98: (98, 2) array of facial landmarks
            frame_num: Frame number (0-indexed)
            timestamp: Timestamp in seconds
            confidence: Detection confidence (0-1)
            success: Success flag (1 or 0)

        Returns:
            dict: Complete row with frame metadata + all AUs
        """
        # Get adapted AUs
        adapted_aus = self.adapt_frame(au_vector_8d, landmarks_98, frame_num)

        # Create CSV row with metadata first, then AUs in correct order
        csv_row = {
            'frame': frame_num,
            'face_id': 0,  # Always 0 for single face tracking
            'timestamp': timestamp,
            'confidence': confidence,
            'success': success,
        }

        # Add intensity AUs (_r) in correct order
        for au_name in self.expected_aus_r:
            csv_row[au_name] = adapted_aus[au_name]

        # Add binary AUs (_c) in correct order
        for au_name in self.expected_aus_c:
            csv_row[au_name] = adapted_aus[au_name]

        return csv_row

    def reset(self):
        """
        Reset internal state (e.g., AU45 smoothing history)

        Call this between videos to ensure temporal smoothing doesn't
        carry over between different video sessions.
        """
        self.au45_calculator.reset()

    def get_au_availability_report(self):
        """
        Get a report of which AUs are available vs missing

        Returns:
            dict: Report with available, missing, and calculated AUs
        """
        return {
            'available_aus': sorted(list(self.available_aus)),
            'missing_aus': sorted(list(self.missing_aus)),
            'calculated_aus': ['AU45_r'],
            'total_available': len(self.available_aus) + 1,  # +1 for AU45
            'total_missing': len(self.missing_aus),
            'total_expected': len(self.expected_aus_r)
        }


def test_adapter():
    """Test function to verify adapter functionality"""
    print("Testing OpenFace3To18AUAdapter...")
    print("="*60)

    # Create adapter
    adapter = OpenFace3To18AUAdapter()

    # Print AU availability report
    report = adapter.get_au_availability_report()
    print("\nAU Availability Report:")
    print(f"  Available AUs: {report['available_aus']}")
    print(f"  Calculated AUs: {report['calculated_aus']}")
    print(f"  Missing AUs: {report['missing_aus']}")
    print(f"  Total: {report['total_available']}/{report['total_expected']} available")

    # Create sample OpenFace 3.0 output (8 AUs with random values 0-5)
    sample_au_8d = np.random.rand(8) * 5.0
    print(f"\nSample OpenFace 3.0 output (8 AUs):")
    for idx, au_name in adapter.of3_au_mapping.items():
        print(f"  {au_name}: {sample_au_8d[idx]:.2f}")

    # Create sample landmarks (98 points)
    sample_landmarks = np.random.rand(98, 2) * 100

    # Test adaptation
    adapted = adapter.adapt_frame(sample_au_8d, sample_landmarks, frame_num=0)

    print(f"\nAdapted output (18 AUs):")
    for au_name in adapter.expected_aus_r:
        value = adapted[au_name]
        if np.isnan(value):
            print(f"  {au_name}: NaN (missing)")
        else:
            print(f"  {au_name}: {value:.2f}")

    print("\n" + "="*60)
    print("Adapter test complete!")


if __name__ == "__main__":
    test_adapter()
