# rthook-transformers.py - Minimal runtime hook for PyInstaller compatibility
# This runs before any user code
#
# NOTE: This hook intentionally does NOT import transformers or torch!
# Those imports trigger the full import chain which fails during the runtime hook phase.
# Instead, we only patch lightweight stdlib modules here.

import sys
import os
from pathlib import Path

# Logging
try:
    hook_log = Path.home() / "Documents" / "SplitFace" / "S2_Logs" / "rthook.log"
    hook_log.parent.mkdir(parents=True, exist_ok=True)
    with open(hook_log, 'w') as f:
        f.write("Runtime hook started\n")
except:
    pass

# Environment variables for transformers
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Patch importlib.metadata to handle missing packages (safe - no heavy imports)
def patch_importlib_metadata():
    try:
        import importlib.metadata as metadata
        original_version = metadata.version
        def patched_version(package_name):
            try:
                return original_version(package_name)
            except metadata.PackageNotFoundError:
                return "0.0.0"
        metadata.version = patched_version
        print("Runtime hook: Patched importlib.metadata.version")
    except Exception as e:
        print(f"Runtime hook: Could not patch importlib.metadata: {e}")

patch_importlib_metadata()

# Patch inspect.getfile for transformers path compatibility (safe - no heavy imports)
import inspect
import traceback as tb
_original_getfile = inspect.getfile

def _safe_getfile(obj):
    try:
        path = _original_getfile(obj)
        stack = tb.extract_stack()
        is_from_transformers = any('transformers' in frame.filename and 'auto_docstring' in frame.filename
                                   for frame in stack)
        if is_from_transformers:
            parts = path.split(os.path.sep)
            if len(parts) < 4:
                return os.path.sep.join(['', 'fake', 'transformers', 'models', 'model', 'module.py'])
        return path
    except (TypeError, OSError):
        return _original_getfile(obj)

inspect.getfile = _safe_getfile
print("Runtime hook: Patched inspect.getfile")

# Pre-install a stub whisperx.vads.pyannote module to prevent pyannote.audio import
# This must happen BEFORE whisperx is imported
import types

# Create a stub Pyannote class
class StubPyannote:
    """Stub class - Pyannote VAD is not available in this build."""
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "Pyannote VAD is not available in this build. "
            "Please use Silero VAD instead (vad_method='silero')."
        )

    @staticmethod
    def preprocess_audio(audio):
        raise ImportError("Pyannote VAD is not available in this build.")

    @staticmethod
    def merge_chunks(*args, **kwargs):
        raise ImportError("Pyannote VAD is not available in this build.")

# Create a fake pyannote submodule to prevent the real one from loading
fake_pyannote_vad = types.ModuleType('whisperx.vads.pyannote')
fake_pyannote_vad.Pyannote = StubPyannote
sys.modules['whisperx.vads.pyannote'] = fake_pyannote_vad

# Create comprehensive pyannote stubs to allow whisperx to import
# without triggering pytorch_lightning dependency

# Stub Pipeline class that whisperx checks for
class StubPipeline:
    """Stub Pipeline class - pyannote is not available in this build."""
    def __init__(self, *args, **kwargs):
        raise ImportError("pyannote.audio is not available in this build.")

# Create the pyannote module hierarchy as proper packages (with __path__)
fake_pyannote = types.ModuleType('pyannote')
fake_pyannote.__path__ = []  # Makes it a package
fake_pyannote.__package__ = 'pyannote'

# Stub Annotation and other pyannote.core classes
class StubAnnotation:
    """Stub Annotation class - pyannote is not available in this build."""
    pass

class StubSegment:
    """Stub Segment class."""
    pass

class StubTimeline:
    """Stub Timeline class."""
    pass

fake_pyannote_core = types.ModuleType('pyannote.core')
fake_pyannote_core.__path__ = []
fake_pyannote_core.__package__ = 'pyannote.core'
fake_pyannote_core.Annotation = StubAnnotation
fake_pyannote_core.Segment = StubSegment
fake_pyannote_core.Timeline = StubTimeline

fake_pyannote_audio = types.ModuleType('pyannote.audio')
fake_pyannote_audio.__path__ = []
fake_pyannote_audio.__package__ = 'pyannote.audio'
fake_pyannote_audio.Pipeline = StubPipeline

fake_pyannote_audio_core = types.ModuleType('pyannote.audio.core')
fake_pyannote_audio_core.__path__ = []
fake_pyannote_audio_core.__package__ = 'pyannote.audio.core'

fake_pyannote_audio_core_model = types.ModuleType('pyannote.audio.core.model')
fake_pyannote_audio_core_inference = types.ModuleType('pyannote.audio.core.inference')
fake_pyannote_audio_core_inference.Inference = StubPipeline
fake_pyannote_audio_pipelines = types.ModuleType('pyannote.audio.pipelines')
fake_pyannote_audio_pipelines.__path__ = []
fake_pyannote_audio_pipelines.__package__ = 'pyannote.audio.pipelines'
fake_pyannote_audio_pipelines_utils = types.ModuleType('pyannote.audio.pipelines.utils')

# Pre-register all fake modules
sys.modules['pyannote'] = fake_pyannote
sys.modules['pyannote.core'] = fake_pyannote_core
sys.modules['pyannote.audio'] = fake_pyannote_audio
sys.modules['pyannote.audio.core'] = fake_pyannote_audio_core
sys.modules['pyannote.audio.core.model'] = fake_pyannote_audio_core_model
sys.modules['pyannote.audio.core.inference'] = fake_pyannote_audio_core_inference
sys.modules['pyannote.audio.pipelines'] = fake_pyannote_audio_pipelines
sys.modules['pyannote.audio.pipelines.utils'] = fake_pyannote_audio_pipelines_utils
print("Runtime hook: Pre-installed comprehensive pyannote stubs")

print("Runtime hook: Completed (minimal patches only)")
try:
    with open(hook_log, 'a') as f:
        f.write("Runtime hook completed\n")
except:
    pass
