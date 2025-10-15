"""
Platform-agnostic path configuration for Action Coder (S2)
Handles file paths for both development and bundled (PyInstaller) environments
Works on both Windows and macOS
"""
import sys
import os
import shutil
from pathlib import Path

# Version information
VERSION = "2.0.0"
APP_NAME = "Action Coder"

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
    Get S2 output base directory in user's Documents folder
    Creates directory structure if it doesn't exist

    Returns:
        Path: ~/Documents/SplitFace/S2O Coded Files/
    """
    docs = get_documents_dir()
    output_base = docs / "SplitFace" / "S2O Coded Files"
    output_base.mkdir(parents=True, exist_ok=True)
    return output_base


def get_resource_path(relative_path):
    """
    Get absolute path to resource (works in both dev and bundled)

    Args:
        relative_path: Relative path to resource (e.g., 'bin/ffmpeg')

    Returns:
        Path: Absolute path to resource
    """
    base_path = get_app_dir()
    resource = base_path / relative_path

    # Note: Not verifying existence here as resources might be optional
    return resource


def get_ffmpeg_path():
    """
    Find FFmpeg executable (cross-platform, bundled or system)

    Search order:
    1. Bundled FFmpeg in app (PyInstaller bundle)
    2. System FFmpeg (from PATH)
    3. macOS Homebrew default location

    Returns:
        str: Path to FFmpeg executable, or None if not found
    """
    if getattr(sys, 'frozen', False):
        # Running in bundle - look in bundled bin directory
        app_dir = get_app_dir()
        if sys.platform == 'win32':
            bundled_ffmpeg = app_dir / 'bin' / 'ffmpeg.exe'
        else:
            bundled_ffmpeg = app_dir / 'bin' / 'ffmpeg'

        if bundled_ffmpeg.exists():
            # Ensure it's executable on Unix-like systems
            if sys.platform != 'win32':
                import stat
                bundled_ffmpeg.chmod(bundled_ffmpeg.stat().st_mode | stat.S_IEXEC)
            return str(bundled_ffmpeg)

    # Development or system FFmpeg
    # Try to find in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path

    # macOS: Try Homebrew default locations
    if sys.platform == 'darwin':
        homebrew_paths = [
            '/opt/homebrew/bin/ffmpeg',  # Apple Silicon
            '/usr/local/bin/ffmpeg',      # Intel Mac
        ]
        for path in homebrew_paths:
            if Path(path).exists():
                return path

    # Not found
    return None


def get_whisper_cache_dir():
    """
    Get platform-specific cache directory for Whisper models

    Returns:
        Path: Cache directory for faster-whisper models
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
    print(f"Action Coder v{VERSION} - Path Configuration")
    print(f"{'='*60}")
    print(f"Running Mode: {'BUNDLED' if is_frozen() else 'DEVELOPMENT'}")
    print(f"Platform: {sys.platform}")
    print(f"\nDirectories:")
    print(f"  App Dir:           {get_app_dir()}")
    print(f"  Documents:         {get_documents_dir()}")
    print(f"  Output Base:       {get_output_base_dir()}")
    print(f"  Whisper Cache:     {get_whisper_cache_dir()}")
    print(f"\nExecutables:")
    ffmpeg = get_ffmpeg_path()
    print(f"  FFmpeg:            {ffmpeg if ffmpeg else 'NOT FOUND'}")
    print(f"{'='*60}\n")


# For testing
if __name__ == "__main__":
    print_paths_info()
