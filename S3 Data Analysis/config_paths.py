"""
Platform-agnostic path configuration for Data Analysis (S3)
Handles file paths for both development and bundled (PyInstaller) environments
Works on both Windows and macOS
"""
import sys
import os
from pathlib import Path

# Version information
VERSION = "1.0.0"
APP_NAME = "Data Analysis"

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


def get_s2_coded_dir():
    """
    Get S2 coded files directory (input for S3)

    Returns:
        Path: ~/Documents/SplitFace/S2O Coded Files/
    """
    docs = get_documents_dir()
    s2_coded = docs / "SplitFace" / "S2O Coded Files"
    return s2_coded


def get_output_base_dir():
    """
    Get S3 output base directory in user's Documents folder
    Creates directory structure if it doesn't exist

    Returns:
        Path: ~/Documents/SplitFace/S3O Results/
    """
    docs = get_documents_dir()
    output_base = docs / "SplitFace" / "S3O Results"
    output_base.mkdir(parents=True, exist_ok=True)
    return output_base


def get_resource_path(relative_path):
    """
    Get absolute path to resource (works in both dev and bundled)

    Args:
        relative_path: Relative path to resource (e.g., 'models/upper_face_model.pkl')

    Returns:
        Path: Absolute path to resource
    """
    base_path = get_app_dir()
    resource = base_path / relative_path

    # Verify resource exists (critical for model files)
    if not resource.exists():
        raise FileNotFoundError(
            f"Resource not found: {relative_path}\n"
            f"Expected at: {resource}\n"
            f"App directory: {base_path}"
        )

    return resource


def get_models_dir():
    """
    Get path to models directory containing ML model files

    Returns:
        Path: Path to models/ directory
    """
    return get_resource_path('models')


def get_model_file(model_name):
    """
    Get path to a specific model file

    Args:
        model_name: Name of model file (e.g., 'upper_face_model.pkl')

    Returns:
        Path: Full path to model file

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path = get_models_dir() / model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_name}\n"
            f"Expected at: {model_path}\n"
            f"Models directory: {get_models_dir()}"
        )
    return model_path


def list_available_models():
    """
    List all available model files in the models directory

    Returns:
        list: List of model filenames
    """
    try:
        models_dir = get_models_dir()
        return [f.name for f in models_dir.glob('*.pkl')] + \
               [f.name for f in models_dir.glob('*.joblib')]
    except FileNotFoundError:
        return []


def is_frozen():
    """
    Check if running in a PyInstaller bundle

    Returns:
        bool: True if frozen (bundled), False if running from source
    """
    return getattr(sys, 'frozen', False)


def print_paths_info():
    """Debug function to print all configured paths"""
    print(f"\n{'='*60}")
    print(f"Data Analysis v{VERSION} - Path Configuration")
    print(f"{'='*60}")
    print(f"Running Mode: {'BUNDLED' if is_frozen() else 'DEVELOPMENT'}")
    print(f"Platform: {sys.platform}")
    print(f"\nDirectories:")
    print(f"  App Dir:           {get_app_dir()}")
    print(f"  Documents:         {get_documents_dir()}")
    print(f"  Output Base:       {get_output_base_dir()}")

    try:
        print(f"  Models Dir:        {get_models_dir()}")
        models = list_available_models()
        if models:
            print(f"\nAvailable Models ({len(models)}):")
            for model in sorted(models):
                print(f"    • {model}")
        else:
            print(f"\nAvailable Models: None found")
    except FileNotFoundError:
        print(f"  Models Dir:        NOT FOUND")

    print(f"{'='*60}\n")


# For testing
if __name__ == "__main__":
    print_paths_info()
