"""
Platform-agnostic path configuration for Face Mirror (S1)
Handles file paths for both development and bundled (PyInstaller) environments
Works on both Windows and macOS
"""
import sys
import os
from pathlib import Path

# Version information
VERSION = "2.0.0"
APP_NAME = "Face Mirror"

def get_app_dir():
    """
    Get the application directory (works in both dev and bundled)

    Returns:
        Path: Application directory containing code and resources
    """
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        return Path(sys._MEIPASS)
    else:
        # Running in development
        return Path(__file__).parent


def get_documents_dir():
    """
    Get the user's Documents folder (cross-platform)

    Returns:
        Path: User's Documents directory
            Windows: C:\\Users\\Username\\Documents
            macOS: /Users/Username/Documents
    """
    if sys.platform == 'win32':
        # Windows: Use Shell API to get Documents folder
        try:
            import ctypes.wintypes
            CSIDL_PERSONAL = 5  # My Documents
            SHGFP_TYPE_CURRENT = 0  # Current value
            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(
                None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf
            )
            return Path(buf.value)
        except Exception:
            # Fallback to environment variable
            return Path(os.path.expanduser("~")) / "Documents"
    else:
        # macOS/Linux: ~/Documents
        return Path.home() / "Documents"


def get_output_base_dir():
    """
    Get S1 output base directory in user's Documents folder
    Creates directory structure if it doesn't exist

    Returns:
        Path: ~/Documents/SplitFace/S1O Processed Files/
    """
    docs = get_documents_dir()
    output_base = docs / "SplitFace" / "S1O Processed Files"
    output_base.mkdir(parents=True, exist_ok=True)
    return output_base


def get_mirror_output_dir():
    """
    Get Face Mirror 1.0 Output directory for mirrored videos

    Returns:
        Path: ~/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/
    """
    base = get_output_base_dir()
    mirror_dir = base / "Face Mirror 1.0 Output"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    return mirror_dir


def get_combined_data_dir():
    """
    Get Combined Data directory for OpenFace CSVs and source videos

    Returns:
        Path: ~/Documents/SplitFace/S1O Processed Files/Combined Data/
    """
    base = get_output_base_dir()
    combined_dir = base / "Combined Data"
    combined_dir.mkdir(parents=True, exist_ok=True)
    return combined_dir


def get_resource_path(relative_path):
    """
    Get absolute path to resource (works in both dev and bundled)

    Args:
        relative_path: Relative path to resource (e.g., 'weights/model.pth')

    Returns:
        Path: Absolute path to resource
    """
    base_path = get_app_dir()
    resource = base_path / relative_path

    # Verify resource exists
    if not resource.exists():
        raise FileNotFoundError(
            f"Resource not found: {relative_path}\n"
            f"Expected at: {resource}\n"
            f"App directory: {base_path}"
        )

    return resource


def get_weights_dir():
    """
    Get path to weights directory containing model files

    Returns:
        Path: Path to weights/ directory
    """
    return get_resource_path('weights')


def is_frozen():
    """
    Check if running in a PyInstaller bundle

    Returns:
        bool: True if frozen (bundled), False if running from source
    """
    return getattr(sys, 'frozen', False)


def get_cache_dir():
    """
    Get platform-specific cache directory for downloaded models

    Returns:
        Path: Cache directory for Hugging Face models
            Windows: C:\\Users\\Username\\.cache\\huggingface\\hub\\
            macOS: /Users/Username/.cache/huggingface/hub/
    """
    if sys.platform == 'win32':
        cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        cache_base = Path.home() / '.cache'

    cache_dir = cache_base / 'huggingface' / 'hub'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def print_paths_info():
    """Debug function to print all configured paths"""
    print(f"\n{'='*60}")
    print(f"Face Mirror v{VERSION} - Path Configuration")
    print(f"{'='*60}")
    print(f"Running Mode: {'BUNDLED' if is_frozen() else 'DEVELOPMENT'}")
    print(f"Platform: {sys.platform}")
    print(f"\nDirectories:")
    print(f"  App Dir:           {get_app_dir()}")
    print(f"  Documents:         {get_documents_dir()}")
    print(f"  Output Base:       {get_output_base_dir()}")
    print(f"  Mirror Output:     {get_mirror_output_dir()}")
    print(f"  Combined Data:     {get_combined_data_dir()}")
    print(f"  Weights:           {get_weights_dir()}")
    print(f"  HF Cache:          {get_cache_dir()}")
    print(f"{'='*60}\n")


# For testing
if __name__ == "__main__":
    print_paths_info()
