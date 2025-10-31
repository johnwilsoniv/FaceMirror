"""
Hardware Detection Module for Paralysis Training Pipeline

Detects system hardware capabilities and provides optimal configuration
for XGBoost training and SHAP analysis.
"""

import platform
import psutil
import logging
import sys

logger = logging.getLogger(__name__)


def detect_hardware():
    """
    Auto-detect hardware capabilities for optimal training configuration.

    Returns:
        dict: Hardware information including:
            - processor: CPU name/type
            - architecture: CPU architecture (arm64, x86_64, etc.)
            - cpu_cores_physical: Number of physical CPU cores
            - cpu_cores_logical: Number of logical CPU cores (with hyperthreading)
            - memory_gb: Total system memory in GB
            - is_apple_silicon: True if running on Apple M1/M2/M3
            - is_macos: True if running on macOS
            - is_windows: True if running on Windows
            - is_linux: True if running on Linux
            - cuda_available: True if CUDA GPU is available for XGBoost
            - recommended_n_jobs: Recommended number of parallel jobs
            - recommended_tree_method: Recommended XGBoost tree method
    """

    try:
        # Basic system info
        hw_info = {
            'processor': platform.processor() or 'Unknown',
            'architecture': platform.machine(),
            'system': platform.system(),
            'python_version': platform.python_version(),
        }

        # CPU information
        try:
            hw_info['cpu_cores_physical'] = psutil.cpu_count(logical=False)
            hw_info['cpu_cores_logical'] = psutil.cpu_count(logical=True)
        except Exception as e:
            logger.warning(f"Could not detect CPU cores: {e}")
            hw_info['cpu_cores_physical'] = 1
            hw_info['cpu_cores_logical'] = 1

        # Memory information
        try:
            hw_info['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception as e:
            logger.warning(f"Could not detect memory: {e}")
            hw_info['memory_gb'] = 0

        # Platform detection
        hw_info['is_macos'] = hw_info['system'] == 'Darwin'
        hw_info['is_windows'] = hw_info['system'] == 'Windows'
        hw_info['is_linux'] = hw_info['system'] == 'Linux'

        # Apple Silicon detection
        hw_info['is_apple_silicon'] = (
            hw_info['architecture'] == 'arm64' and hw_info['is_macos']
        )

        # CUDA availability check
        hw_info['cuda_available'] = check_cuda_available()

        # Generate recommendations
        hw_info['recommended_n_jobs'] = get_recommended_n_jobs(hw_info)
        hw_info['recommended_tree_method'] = get_recommended_tree_method(hw_info)
        hw_info['recommended_optuna_jobs'] = get_recommended_optuna_jobs(hw_info)

        logger.info(f"Hardware detected: {hw_info['processor']} ({hw_info['architecture']}) "
                   f"with {hw_info['cpu_cores_physical']} cores, {hw_info['memory_gb']} GB RAM")

        return hw_info

    except Exception as e:
        logger.error(f"Error detecting hardware: {e}", exc_info=True)
        # Return safe defaults
        return {
            'processor': 'Unknown',
            'architecture': 'Unknown',
            'cpu_cores_physical': 1,
            'cpu_cores_logical': 1,
            'memory_gb': 0,
            'is_apple_silicon': False,
            'is_macos': False,
            'is_windows': False,
            'is_linux': False,
            'cuda_available': False,
            'recommended_n_jobs': 1,
            'recommended_tree_method': 'hist',
            'recommended_optuna_jobs': 1,
        }


def check_cuda_available():
    """
    Check if CUDA GPU is available for XGBoost.

    Returns:
        bool: True if CUDA is available and working
    """
    try:
        import xgboost as xgb
        import numpy as np

        # Try to create a small DMatrix on GPU
        test_data = np.random.rand(10, 5)
        test_dmatrix = xgb.DMatrix(test_data)

        # Try to set device to CUDA
        test_dmatrix.set_info(device='cuda')

        logger.info("CUDA GPU detected and available for XGBoost")
        return True

    except Exception as e:
        logger.debug(f"CUDA not available: {e}")
        return False


def get_recommended_n_jobs(hw_info):
    """
    Get recommended number of parallel jobs for XGBoost training.

    Args:
        hw_info (dict): Hardware information from detect_hardware()

    Returns:
        int: Recommended n_jobs value (-1 for all cores, or specific number)
    """
    # If GPU available, use 1 job (GPU handles parallelism)
    if hw_info['cuda_available']:
        return 1

    # For CPU: use all available cores
    # Using logical cores (with hyperthreading) is generally good for tree methods
    cores = hw_info.get('cpu_cores_logical', 1)

    if cores >= 4:
        return -1  # Use all cores
    else:
        return cores  # Use specific number for low-core systems


def get_recommended_tree_method(hw_info):
    """
    Get recommended tree_method for XGBoost based on hardware.

    Args:
        hw_info (dict): Hardware information from detect_hardware()

    Returns:
        str: Recommended tree_method ('hist', 'gpu_hist', 'approx', 'exact')
    """
    # GPU available: use gpu_hist (fastest for large datasets)
    if hw_info['cuda_available']:
        return 'gpu_hist'

    # CPU: use hist (histogram-based, fast and memory efficient)
    # 'hist' is optimal for most cases and is the modern default
    return 'hist'


def get_recommended_optuna_jobs(hw_info):
    """
    Get recommended number of parallel Optuna trials.

    Args:
        hw_info (dict): Hardware information from detect_hardware()

    Returns:
        int: Recommended parallel trials (1-8)
    """
    cores = hw_info.get('cpu_cores_physical', 1)

    if cores >= 8:
        return 4  # 4 parallel trials for 8+ cores
    elif cores >= 4:
        return 2  # 2 parallel trials for 4-7 cores
    else:
        return 1  # Sequential for <4 cores


def get_xgboost_params(hw_info, base_params=None):
    """
    Get optimized XGBoost parameters based on detected hardware.

    Args:
        hw_info (dict): Hardware information from detect_hardware()
        base_params (dict): Base parameters to update (optional)

    Returns:
        dict: XGBoost parameters optimized for hardware
    """
    if base_params is None:
        base_params = {}

    # Update with hardware-specific settings
    hw_params = {
        'tree_method': hw_info['recommended_tree_method'],
        'n_jobs': hw_info['recommended_n_jobs'],
    }

    # Add device specification for GPU
    if hw_info['cuda_available']:
        hw_params['device'] = 'cuda'

    # Merge with base params (base params take precedence for non-hardware settings)
    optimized_params = {**hw_params, **base_params}

    return optimized_params


def format_hardware_info(hw_info):
    """
    Format hardware information for display in GUI or logs.

    Args:
        hw_info (dict): Hardware information from detect_hardware()

    Returns:
        str: Formatted hardware information string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("HARDWARE CONFIGURATION")
    lines.append("=" * 60)

    # Processor info
    lines.append(f"Processor: {hw_info['processor']}")
    lines.append(f"Architecture: {hw_info['architecture']}")

    if hw_info['is_apple_silicon']:
        lines.append("Platform: Apple Silicon (M1/M2/M3)")
    elif hw_info['is_macos']:
        lines.append("Platform: macOS (Intel)")
    elif hw_info['is_windows']:
        lines.append("Platform: Windows")
    elif hw_info['is_linux']:
        lines.append("Platform: Linux")

    # CPU and memory
    lines.append(f"CPU Cores: {hw_info['cpu_cores_physical']} physical, "
                f"{hw_info['cpu_cores_logical']} logical")
    lines.append(f"Memory: {hw_info['memory_gb']} GB")

    # GPU info
    if hw_info['cuda_available']:
        lines.append("GPU: CUDA Available ✓")
    else:
        lines.append("GPU: Not Available (CPU-only mode)")
        if hw_info['is_apple_silicon']:
            lines.append("  Note: Apple Silicon GPU not supported by XGBoost")

    lines.append("")
    lines.append("RECOMMENDED SETTINGS")
    lines.append("-" * 60)
    lines.append(f"XGBoost tree_method: {hw_info['recommended_tree_method']}")
    lines.append(f"XGBoost n_jobs: {hw_info['recommended_n_jobs']} "
                f"({'all cores' if hw_info['recommended_n_jobs'] == -1 else 'cores'})")
    lines.append(f"Optuna parallel trials: {hw_info['recommended_optuna_jobs']}")
    lines.append("=" * 60)

    return "\n".join(lines)


def check_fasttreeshap_available():
    """
    Check if FastTreeSHAP is installed and available.

    Returns:
        bool: True if FastTreeSHAP is available
    """
    try:
        import fasttreeshap
        logger.info("FastTreeSHAP is available (1.5-2.7x faster SHAP computation)")
        return True
    except ImportError:
        logger.warning("FastTreeSHAP not available, falling back to standard SHAP TreeExplainer")
        logger.warning("Install with: pip install fasttreeshap")
        return False


if __name__ == "__main__":
    # Test hardware detection
    import logging
    logging.basicConfig(level=logging.INFO)

    hw_info = detect_hardware()
    print(format_hardware_info(hw_info))

    print("\nFastTreeSHAP available:", check_fasttreeshap_available())

    print("\nOptimized XGBoost params:")
    params = get_xgboost_params(hw_info)
    for key, value in params.items():
        print(f"  {key}: {value}")
