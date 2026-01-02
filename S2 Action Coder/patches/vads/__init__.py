# Patched whisperx/vads/__init__.py - Makes Pyannote completely unavailable
# Pyannote requires pyannote.audio which requires pytorch_lightning
# Since we use Silero VAD, we don't need Pyannote at all

from whisperx.vads.silero import Silero as Silero
from whisperx.vads.vad import Vad as Vad

# DO NOT import pyannote - it will fail due to missing pytorch_lightning
# Instead, provide a stub class that raises a helpful error
class Pyannote:
    """Stub class - Pyannote VAD is not available in this build."""
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "Pyannote VAD is not available in this build. "
            "Please use Silero VAD instead (vad_method='silero')."
        )
