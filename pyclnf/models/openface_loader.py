"""
OpenFace Model Loader - Parse OpenFace models directly without C++ dependency

This module provides utilities to load OpenFace PDM and patch expert models
directly from their native text/binary formats.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import struct


class PDMLoader:
    """Load and parse OpenFace PDM (Point Distribution Model) from text file."""

    def __init__(self, model_path: str):
        """
        Load PDM from OpenFace text format.

        Args:
            model_path: Path to PDM .txt file (e.g., In-the-wild_aligned_PDM_68.txt)
        """
        self.model_path = Path(model_path)
        self.mean_shape = None
        self.princ_comp = None  # Principal components
        self.eigen_values = None

        self._load()

    def _skip_comments(self, f):
        """Skip lines starting with # and empty lines."""
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # Not a comment, go back
                f.seek(pos)
                return

    def _read_matrix(self, f) -> np.ndarray:
        """
        Read a matrix in OpenFace text format:
        Line 1: rows
        Line 2: cols
        Line 3: type (OpenCV type code)
        Remaining lines: data values (one row per line, space-separated)
        """
        self._skip_comments(f)
        rows = int(f.readline().strip())

        self._skip_comments(f)
        cols = int(f.readline().strip())

        self._skip_comments(f)
        cv_type = int(f.readline().strip())

        # CV_64FC1 = 6, CV_32FC1 = 5
        # Determine data type from OpenCV type code
        if cv_type == 6:  # CV_64FC1
            dtype = np.float64
        elif cv_type == 5:  # CV_32FC1
            dtype = np.float32
        else:
            dtype = np.float64  # Default to double

        # Read data values (one row per line, space-separated)
        data = []
        for _ in range(rows):
            self._skip_comments(f)
            line = f.readline().strip()
            # Split by whitespace and convert to floats
            row_values = [float(x) for x in line.split()]
            if len(row_values) != cols:
                raise ValueError(f"Expected {cols} values, got {len(row_values)}")
            data.extend(row_values)

        # Reshape to matrix
        matrix = np.array(data, dtype=dtype).reshape(rows, cols)
        return matrix

    def _load(self):
        """Load all PDM components from file."""
        with open(self.model_path, 'r') as f:
            # Read mean shape
            self.mean_shape = self._read_matrix(f)

            # Read principal components
            self.princ_comp = self._read_matrix(f)

            # Read eigenvalues
            self.eigen_values = self._read_matrix(f)

    def save_numpy(self, output_dir: str):
        """
        Export PDM to NumPy .npy files.

        Args:
            output_dir: Directory to save .npy files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.save(output_path / 'mean_shape.npy', self.mean_shape.astype(np.float32))
        np.save(output_path / 'princ_comp.npy', self.princ_comp.astype(np.float32))
        np.save(output_path / 'eigen_values.npy', self.eigen_values.astype(np.float32))

        print(f"PDM exported to {output_path}")
        print(f"  - mean_shape.npy: shape {self.mean_shape.shape}")
        print(f"  - princ_comp.npy: shape {self.princ_comp.shape}")
        print(f"  - eigen_values.npy: shape {self.eigen_values.shape}")

    def number_of_points(self) -> int:
        """Return number of landmarks (should be 68 for face)."""
        return self.mean_shape.shape[0] // 3

    def number_of_modes(self) -> int:
        """Return number of PCA modes."""
        return self.princ_comp.shape[1]

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'num_points': self.number_of_points(),
            'num_modes': self.number_of_modes(),
            'mean_shape_shape': self.mean_shape.shape,
            'princ_comp_shape': self.princ_comp.shape,
            'eigen_values_shape': self.eigen_values.shape,
        }


class CCNFPatchExpertLoader:
    """Load and parse OpenFace CCNF patch experts from binary file."""

    def __init__(self, model_path: str):
        """
        Load CCNF patch experts from OpenFace binary format.

        Args:
            model_path: Path to CCNF .txt file (e.g., ccnf_patches_0.25_general.txt)
        """
        self.model_path = Path(model_path)
        self.patches = []  # List of patch experts (one per landmark)

        self._load()

    def _read_int32(self, f) -> int:
        """Read 4-byte integer."""
        return struct.unpack('i', f.read(4))[0]

    def _read_float64(self, f) -> float:
        """Read 8-byte double."""
        return struct.unpack('d', f.read(8))[0]

    def _read_matrix_bin(self, f) -> np.ndarray:
        """
        Read binary matrix in OpenFace format:
        - 4 bytes: rows (int)
        - 4 bytes: cols (int)
        - 4 bytes: type (OpenCV type code)
        - remaining: data
        """
        rows = self._read_int32(f)
        cols = self._read_int32(f)
        cv_type = self._read_int32(f)

        # Determine dtype and element size
        if cv_type == 5:  # CV_32FC1
            dtype = np.float32
            elem_size = 4
        elif cv_type == 6:  # CV_64FC1
            dtype = np.float64
            elem_size = 8
        else:
            raise ValueError(f"Unsupported OpenCV type: {cv_type}")

        # Read data
        num_elements = rows * cols
        data = np.frombuffer(f.read(num_elements * elem_size), dtype=dtype)

        return data.reshape(rows, cols)

    def _read_neuron(self, f) -> Dict[str, Any]:
        """Read a single CCNF neuron from binary file."""
        # Read type marker (should be 2)
        read_type = self._read_int32(f)
        if read_type != 2:
            raise ValueError(f"Expected neuron type 2, got {read_type}")

        # Read neuron parameters
        neuron_type = self._read_int32(f)
        norm_weights = self._read_float64(f)
        bias = self._read_float64(f)
        alpha = self._read_float64(f)

        # Read weight matrix
        weights = self._read_matrix_bin(f)

        return {
            'neuron_type': neuron_type,
            'norm_weights': norm_weights,
            'bias': bias,
            'alpha': alpha,
            'weights': weights
        }

    def _read_patch_expert(self, f) -> Dict[str, Any]:
        """Read a single patch expert (for one landmark)."""
        # This is a simplified version - full implementation would read all metadata
        # For now, we'll read the neurons which are the core components

        neurons = []
        # Need to determine how many neurons - this depends on the file format
        # For now, read until we hit an expected marker or EOF

        return {
            'neurons': neurons
        }

    def _load(self):
        """Load CCNF patch experts from binary file."""
        # Note: The exact binary format is complex and depends on how OpenFace stores it
        # This is a placeholder - full implementation requires careful analysis of
        # CCNF_patch_expert.cpp Read() method

        with open(self.model_path, 'rb') as f:
            # Parse binary format
            # This will be implemented after analyzing the exact file structure
            pass

    def save_numpy(self, output_dir: str):
        """Export patch experts to NumPy format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export each patch expert
        for i, patch in enumerate(self.patches):
            patch_dir = output_path / f'patch_{i:02d}'
            patch_dir.mkdir(exist_ok=True)

            # Save neurons
            for j, neuron in enumerate(patch['neurons']):
                neuron_file = patch_dir / f'neuron_{j:02d}.npz'
                np.savez(
                    neuron_file,
                    neuron_type=neuron['neuron_type'],
                    norm_weights=neuron['norm_weights'],
                    bias=neuron['bias'],
                    alpha=neuron['alpha'],
                    weights=neuron['weights']
                )


def test_pdm_loader():
    """Test PDM loading functionality."""
    openface_dir = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace"
    pdm_path = openface_dir / "lib/local/LandmarkDetector/model/pdms/In-the-wild_aligned_PDM_68.txt"

    if not pdm_path.exists():
        print(f"PDM file not found: {pdm_path}")
        return

    print("Loading PDM...")
    pdm = PDMLoader(str(pdm_path))

    info = pdm.get_info()
    print("\nPDM Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Export to numpy
    output_dir = "pyclnf/models/exported_pdm"
    pdm.save_numpy(output_dir)

    # Verify by loading
    print("\nVerifying exported files...")
    mean_shape = np.load(f"{output_dir}/mean_shape.npy")
    princ_comp = np.load(f"{output_dir}/princ_comp.npy")
    eigen_values = np.load(f"{output_dir}/eigen_values.npy")

    print(f"  Loaded mean_shape: {mean_shape.shape}, dtype: {mean_shape.dtype}")
    print(f"  Loaded princ_comp: {princ_comp.shape}, dtype: {princ_comp.dtype}")
    print(f"  Loaded eigen_values: {eigen_values.shape}, dtype: {eigen_values.dtype}")

    print("\n✓ PDM loading and export successful!")


if __name__ == "__main__":
    test_pdm_loader()
