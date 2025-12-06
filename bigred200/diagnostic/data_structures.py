"""
Data structures for diagnostic collection.

These dataclasses store per-iteration, per-landmark diagnostic information
for root cause analysis of landmark detection errors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
import pickle
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class LandmarkDiagnostic:
    """Per-landmark diagnostic data for one iteration."""
    landmark_idx: int

    # Position tracking
    current_pos: Tuple[float, float]
    base_pos: Tuple[float, float]
    cpp_reference_pos: Optional[Tuple[float, float]] = None

    # Mean-shift computation chain
    response_map: Optional[np.ndarray] = None  # 11x11, 9x9, or 7x7
    response_map_stats: Optional[Dict] = None  # min, max, mean, std, peak_idx, sharpness

    # Offset computation
    offset_ref: Tuple[float, float] = (0.0, 0.0)  # Offset in reference coords
    dx_dy: Tuple[float, float] = (0.0, 0.0)  # Position in response map [0, window_size)
    clamped: bool = False  # Whether dx/dy were clamped to bounds

    # Mean-shift result
    ms_ref: Tuple[float, float] = (0.0, 0.0)  # Mean-shift in reference coords
    ms_img: Tuple[float, float] = (0.0, 0.0)  # Mean-shift in image coords (final)

    # KDE details
    kde_total_weight: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            'landmark_idx': self.landmark_idx,
            'current_pos': list(self.current_pos),
            'base_pos': list(self.base_pos),
            'offset_ref': list(self.offset_ref),
            'dx_dy': list(self.dx_dy),
            'clamped': self.clamped,
            'ms_ref': list(self.ms_ref),
            'ms_img': list(self.ms_img),
            'kde_total_weight': self.kde_total_weight,
        }
        if self.cpp_reference_pos is not None:
            d['cpp_reference_pos'] = list(self.cpp_reference_pos)
        if self.response_map_stats is not None:
            d['response_map_stats'] = self.response_map_stats
        # Note: response_map itself is saved separately as .npy
        return d


@dataclass
class IterationDiagnostic:
    """Per-iteration diagnostic data for all landmarks."""
    frame_idx: int
    iteration: int
    phase: str  # 'rigid' or 'nonrigid'
    window_size: int

    # Per-landmark diagnostics
    landmarks: Dict[int, LandmarkDiagnostic] = field(default_factory=dict)

    # Global iteration metrics
    update_magnitude: float = 0.0
    mean_shift_norm: float = 0.0
    jacobian_norm: float = 0.0
    hessian_cond: Optional[float] = None
    reg_ratio: Optional[float] = None

    # Transform matrices (for debugging coordinate transforms)
    sim_img_to_ref: Optional[np.ndarray] = None
    sim_ref_to_img: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_idx': self.frame_idx,
            'iteration': self.iteration,
            'phase': self.phase,
            'window_size': self.window_size,
            'landmarks': {k: v.to_dict() for k, v in self.landmarks.items()},
            'update_magnitude': self.update_magnitude,
            'mean_shift_norm': self.mean_shift_norm,
            'jacobian_norm': self.jacobian_norm,
            'hessian_cond': self.hessian_cond,
            'reg_ratio': self.reg_ratio,
        }


@dataclass
class FrameDiagnostic:
    """Complete diagnostic for one frame."""
    frame_idx: int
    video_name: str = ""

    # C++ reference data
    cpp_landmarks: Optional[np.ndarray] = None  # (68, 2)
    cpp_pose: Optional[Tuple[float, float, float]] = None  # rx, ry, rz

    # Python results
    py_landmarks: Optional[np.ndarray] = None  # (68, 2)
    py_pose: Optional[Tuple[float, float, float]] = None
    py_params: Optional[np.ndarray] = None

    # Per-landmark error
    landmark_errors: Optional[np.ndarray] = None  # (68,) Euclidean distances

    # Per-iteration diagnostics
    iterations: List[IterationDiagnostic] = field(default_factory=list)

    # Processing info
    detection_bbox: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    success: bool = True
    error_message: str = ""

    def compute_errors(self):
        """Compute per-landmark errors from py and cpp landmarks."""
        if self.py_landmarks is not None and self.cpp_landmarks is not None:
            self.landmark_errors = np.linalg.norm(
                self.py_landmarks - self.cpp_landmarks, axis=1
            )

    def get_region_errors(self) -> Dict[str, float]:
        """Get mean error per facial region."""
        if self.landmark_errors is None:
            return {}

        regions = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose': list(range(27, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'outer_mouth': list(range(48, 60)),
            'inner_mouth': list(range(60, 68)),
        }

        return {
            region: float(np.mean(self.landmark_errors[indices]))
            for region, indices in regions.items()
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            'frame_idx': self.frame_idx,
            'video_name': self.video_name,
            'processing_time_ms': self.processing_time_ms,
            'success': self.success,
            'error_message': self.error_message,
        }

        if self.py_landmarks is not None:
            d['py_landmarks'] = self.py_landmarks.tolist()
        if self.cpp_landmarks is not None:
            d['cpp_landmarks'] = self.cpp_landmarks.tolist()
        if self.landmark_errors is not None:
            d['landmark_errors'] = self.landmark_errors.tolist()
            d['region_errors'] = self.get_region_errors()
        if self.py_pose is not None:
            d['py_pose'] = [float(x) for x in self.py_pose]
        if self.cpp_pose is not None:
            d['cpp_pose'] = [float(x) for x in self.cpp_pose]
        if self.detection_bbox is not None:
            d['detection_bbox'] = self.detection_bbox.tolist()

        d['iterations'] = [it.to_dict() for it in self.iterations]

        return d

    def save(self, output_dir: Path, save_response_maps: bool = True):
        """Save diagnostic data to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main diagnostic as JSON
        json_path = output_dir / f"frame_{self.frame_idx:05d}.json"
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)

        # Save response maps as numpy arrays (if requested)
        if save_response_maps:
            maps_dir = output_dir / f"frame_{self.frame_idx:05d}_maps"
            maps_dir.mkdir(exist_ok=True)

            for iter_diag in self.iterations:
                for lm_idx, lm_diag in iter_diag.landmarks.items():
                    if lm_diag.response_map is not None:
                        map_path = maps_dir / f"iter{iter_diag.iteration}_lm{lm_idx}_ws{iter_diag.window_size}.npy"
                        np.save(map_path, lm_diag.response_map)

    @classmethod
    def load(cls, json_path: Path) -> 'FrameDiagnostic':
        """Load diagnostic from JSON file."""
        with open(json_path, 'r') as f:
            d = json.load(f)

        diag = cls(
            frame_idx=d['frame_idx'],
            video_name=d.get('video_name', ''),
            processing_time_ms=d.get('processing_time_ms', 0.0),
            success=d.get('success', True),
            error_message=d.get('error_message', ''),
        )

        if 'py_landmarks' in d:
            diag.py_landmarks = np.array(d['py_landmarks'])
        if 'cpp_landmarks' in d:
            diag.cpp_landmarks = np.array(d['cpp_landmarks'])
        if 'landmark_errors' in d:
            diag.landmark_errors = np.array(d['landmark_errors'])
        if 'py_pose' in d:
            diag.py_pose = tuple(d['py_pose'])
        if 'cpp_pose' in d:
            diag.cpp_pose = tuple(d['cpp_pose'])
        if 'detection_bbox' in d:
            diag.detection_bbox = np.array(d['detection_bbox'])

        # Note: iterations are loaded as dicts, not full objects
        # Full loading with response maps requires separate handling

        return diag


def save_all_diagnostics(diagnostics: List[FrameDiagnostic],
                         output_dir: Path,
                         save_response_maps: bool = True):
    """Save all frame diagnostics to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual frames
    for diag in diagnostics:
        diag.save(output_dir, save_response_maps=save_response_maps)

    # Save summary CSV
    summary_rows = []
    for diag in diagnostics:
        if diag.success and diag.landmark_errors is not None:
            row = {
                'frame_idx': diag.frame_idx,
                'mean_error': float(np.mean(diag.landmark_errors)),
                'max_error': float(np.max(diag.landmark_errors)),
                'processing_time_ms': diag.processing_time_ms,
            }
            row.update(diag.get_region_errors())
            summary_rows.append(row)

    if summary_rows:
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        df.to_csv(output_dir / "summary.csv", index=False)
        print(f"Saved {len(summary_rows)} frame diagnostics to {output_dir}")
