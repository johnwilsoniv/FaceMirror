
# whisper_handler.py - Handles audio extraction and Whisper transcription/alignment in a separate thread
import os
import sys
import tempfile
import shutil
import time
import platform
import hashlib
import json
from pathlib import Path

# --- Try MLX Whisper (Apple Silicon optimized) ---
MLX_WHISPER_AVAILABLE = False
IS_APPLE_SILICON = False
try:
    import mlx_whisper
    IS_APPLE_SILICON = platform.machine() == "arm64" and platform.system() == "Darwin"
    if IS_APPLE_SILICON:
        MLX_WHISPER_AVAILABLE = True
        print("Whisper Handler: MLX-Whisper loaded successfully (Apple Silicon optimized)")
        print("  Using Apple Neural Engine for 3-6x speedup with full accuracy")
    else:
        print("Whisper Handler: MLX-Whisper found but not on Apple Silicon, will use faster-whisper")
except ImportError:
    print("Whisper Handler: MLX-Whisper not available, will use faster-whisper")
    MLX_WHISPER_AVAILABLE = False

# --- Import Silero VAD for preprocessing (same as faster-whisper uses) ---
SILERO_VAD_AVAILABLE = False
try:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
    SILERO_VAD_AVAILABLE = True
    print("Whisper Handler: Silero VAD loaded successfully (for speech detection)")
except ImportError as e:
    print(f"Whisper Handler: Silero VAD not available: {e}")
    SILERO_VAD_AVAILABLE = False

# --- Use faster-whisper for transcription (fallback) ---
FASTER_WHISPER_AVAILABLE = False
WhisperModel = None
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    if not MLX_WHISPER_AVAILABLE:
        print("Whisper Handler: faster-whisper library loaded successfully.")
        print("  Using faster-whisper for maximum accuracy with VAD filtering & beam search")
except ImportError as e:
    print(f"Whisper Handler ERROR: faster-whisper import failed: {e}")
    print("  Install with: pip install faster-whisper")
    FASTER_WHISPER_AVAILABLE = False

# --- Keep whisperx ONLY for alignment ---
try:
    import whisperx
    WHISPERX_ALIGN_AVAILABLE = True
    print("Whisper Handler: WhisperX library found (for alignment).")
except ImportError as e:
    print(f"Whisper Handler WARNING: WhisperX import failed: {e}. Alignment step will be skipped.")
    whisperx = None
    WHISPERX_ALIGN_AVAILABLE = False

from PyQt5.QtCore import QThread, pyqtSignal, QObject
import builtins

# --- Try importing torch ---
try:
    import torch
    TORCH_AVAILABLE = True
    print("Whisper Handler: PyTorch found.")
except ImportError:
    TORCH_AVAILABLE = False
    print("Whisper Handler: PyTorch not found, will use CPU.")


# Define InterruptedError if it doesn't exist
if 'InterruptedError' not in builtins.__dict__:
    class InterruptedError(Exception):
        pass


def get_vad_cache_path(audio_path, vad_parameters):
    """
    Generate cache path for VAD results based on audio file and VAD parameters.

    Args:
        audio_path: Path to audio file
        vad_parameters: VAD parameters dict

    Returns:
        Path to cache file
    """
    # Create cache directory
    cache_dir = Path.home() / ".cache" / "action_coder" / "vad"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create unique hash from audio file + VAD params
    audio_stat = os.stat(audio_path)
    cache_key = f"{audio_path}_{audio_stat.st_size}_{audio_stat.st_mtime}_{json.dumps(vad_parameters, sort_keys=True)}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    return cache_dir / f"vad_{cache_hash}.json"


def load_vad_cache(audio_path, vad_parameters):
    """Load cached VAD results if available."""
    cache_path = get_vad_cache_path(audio_path, vad_parameters)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load VAD cache: {e}")
    return None


def save_vad_cache(audio_path, vad_parameters, segments):
    """Save VAD results to cache."""
    cache_path = get_vad_cache_path(audio_path, vad_parameters)
    try:
        with open(cache_path, 'w') as f:
            json.dump(segments, f)
        print(f"VAD results cached: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save VAD cache: {e}")


def run_silero_vad(audio_path, vad_parameters):
    """
    Run Silero VAD to detect speech segments in audio.
    Returns list of speech segments with start/end times in seconds.

    This uses the same Silero VAD model that faster-whisper uses internally,
    but we control it explicitly for MLX Whisper integration.
    """
    if not SILERO_VAD_AVAILABLE:
        raise ImportError("Silero VAD is required but not available. Install with: pip install silero-vad")

    print(f"Silero VAD: Loading audio from {audio_path}")

    # Load VAD model (lightweight 1.8MB model)
    vad_model = load_silero_vad()

    # Read audio file
    audio = read_audio(audio_path, sampling_rate=16000)

    # Get speech timestamps using VAD parameters
    # Convert parameters to Silero VAD format
    threshold = vad_parameters.get("threshold", 0.5)
    min_speech_duration_ms = vad_parameters.get("min_speech_duration_ms", 250)
    min_silence_duration_ms = vad_parameters.get("min_silence_duration_ms", 2000)

    print(f"Silero VAD: threshold={threshold}, min_speech_duration={min_speech_duration_ms}ms, min_silence={min_silence_duration_ms}ms")

    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True  # Return timestamps in seconds
    )

    print(f"Silero VAD: Detected {len(speech_timestamps)} speech segments")

    return speech_timestamps


def transcribe_with_mlx(audio_path, model_name, speech_segments, language="en", beam_size=5):
    """
    Transcribe audio using MLX Whisper with speech segments from Silero VAD.

    Args:
        audio_path: Path to audio file
        model_name: MLX Whisper model (e.g., "mlx-community/whisper-large-v3")
        speech_segments: List of speech segments from Silero VAD
        language: Language code
        beam_size: Beam search size for decoding quality

    Returns:
        List of transcribed segments with text and timestamps
    """
    if not MLX_WHISPER_AVAILABLE:
        raise ImportError("MLX Whisper is required but not available. Install with: pip install mlx-whisper")

    import mlx_whisper

    print(f"MLX Whisper: Transcribing with model {model_name}, beam_size={beam_size}")

    # For large-v3, use the MLX community version
    if "large-v3" in model_name and "mlx-community" not in model_name:
        model_name = "mlx-community/whisper-large-v3-mlx"

    all_segments = []

    # Process each speech segment separately
    for idx, seg in enumerate(speech_segments):
        start_time = seg['start']
        end_time = seg['end']

        print(f"MLX Whisper: Processing segment {idx+1}/{len(speech_segments)}: {start_time:.2f}s - {end_time:.2f}s")

        # Transcribe this segment with MLX Whisper
        # Note: MLX Whisper will process the full file, but we'll use clip_timestamps
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_name,
            language=language,
            verbose=False,
            clip_timestamps=[start_time, end_time],  # Focus on this segment
            condition_on_previous_text=True,
            temperature=0.0,  # Use beam search (no sampling)
            # Note: beam_size is implicit when temperature=0.0 in MLX Whisper
        )

        # Extract segments from result
        segments = result.get("segments", [])

        for segment in segments:
            # Adjust timestamps to be relative to the original audio
            segment_start = segment.get("start", start_time)
            segment_end = segment.get("end", end_time)
            segment_text = segment.get("text", "").strip()

            if segment_text:  # Only add non-empty segments
                all_segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": segment_text
                })

    print(f"MLX Whisper: Transcription complete. Generated {len(all_segments)} segments")

    return all_segments


class WhisperHandler(QThread):
    """
    Runs Whisper transcription + WhisperX alignment in a separate thread.

    On Apple Silicon: Uses MLX Whisper + Silero VAD for 3-6x speedup
    On other platforms: Uses faster-whisper with built-in VAD

    Processes a pre-extracted audio file.
    Returns segments list (with words) on finish.
    """
    progress_update = pyqtSignal(int, str)
    processing_error = pyqtSignal(str)
    processing_finished = pyqtSignal(list) # Emits results list (aligned segments with words)

    audio_path = None
    temp_dir_for_cleanup = None

    # --- MODIFIED __init__ to accept vad_parameters and pre-loaded model ---
    def __init__(self, audio_path, temp_dir_for_cleanup, vad_parameters, model_name="large-v3", parent=None, debug_keep_audio=False, preloaded_model=None, skip_alignment=False):
        super().__init__(parent)
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path does not exist or not provided: {audio_path}")

        # Determine which engine to use
        self.use_mlx = MLX_WHISPER_AVAILABLE and SILERO_VAD_AVAILABLE

        if self.use_mlx:
            print("WhisperHandler: Will use MLX Whisper + Silero VAD (Apple Silicon optimized)")
        else:
            # Check if faster-whisper is available as fallback
            if not FASTER_WHISPER_AVAILABLE:
                 raise ImportError("Neither MLX nor faster-whisper available. Install one with: pip install mlx-whisper silero-vad OR pip install faster-whisper")
            print("WhisperHandler: Will use faster-whisper")

        if not WHISPERX_ALIGN_AVAILABLE:
            print("WhisperHandler WARNING: WhisperX not available, alignment will be skipped.")

        self.audio_path = audio_path
        self.temp_dir_for_cleanup = temp_dir_for_cleanup
        self.model_size_or_path = model_name
        self.vad_parameters = vad_parameters # Store the passed VAD parameters
        self._is_cancelled = False
        self.whisper_model = preloaded_model # Use pre-loaded model if provided
        self.align_model = None
        self.align_metadata = None
        self.debug_keep_audio = debug_keep_audio
        self.skip_alignment = skip_alignment  # Flag to skip word-level alignment
        # print(f"WhisperHandler Initialized with audio: {self.audio_path}, temp_dir: {self.temp_dir_for_cleanup}") # Less verbose
        print(f"WhisperHandler Using VAD Parameters: {self.vad_parameters}") # Log params used
        if preloaded_model:
            print("WhisperHandler: Using pre-loaded Whisper model from splash screen")
        if skip_alignment:
            print("WhisperHandler: Word-level alignment disabled for faster processing")

    # --- MODIFIED run method to support both MLX and faster-whisper ---
    def run(self):
        self._is_cancelled = False
        all_segments = []
        device = "cpu"

        # Check that at least one engine is available
        if not self.use_mlx and not FASTER_WHISPER_AVAILABLE:
            self.processing_error.emit("No transcription engine available.")
            self.processing_finished.emit([])
            return

        try:
            if not self.audio_path or not os.path.exists(self.audio_path):
                 raise FileNotFoundError("Pre-extracted audio path not valid during run.")

            self.emit_progress(10, f"Setting up device...")
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

            # Optimized quantization settings (30-50% speedup with minimal accuracy loss)
            if device == "cuda":
                # int8_float16 provides best speed/accuracy tradeoff on CUDA
                compute_type = "int8_float16"
                print("FasterWhisper: Using int8_float16 quantization on CUDA (optimized)")
            else:
                # int8 works well on CPU
                compute_type = "int8"
                print("FasterWhisper Warning: Running on CPU with int8 quantization.")

            # CPU parallelization settings (1.5-2x speedup on CPU)
            import multiprocessing
            cpu_cores = multiprocessing.cpu_count()
            num_workers = min(4, max(1, cpu_cores // 2))  # Use up to 4 workers, but not more than half the cores
            cpu_threads = max(4, cpu_cores // num_workers)  # Distribute threads among workers

            print(f"FasterWhisper Thread: Using device: {device}, compute_type: {compute_type}")
            if device == "cpu":
                print(f"FasterWhisper Thread: CPU optimization enabled - {num_workers} workers, {cpu_threads} threads per worker")

            # === VAD CACHING: Check cache first ===
            cached_segments = load_vad_cache(self.audio_path, self.vad_parameters)
            if cached_segments:
                print(f"VAD Cache: Found cached segments ({len(cached_segments)} segments)")
                self.emit_progress(50, f"Using cached VAD results...")
                faster_whisper_segments = cached_segments
                # Create mock info object for cached results
                class MockInfo:
                    language = "en"
                    language_probability = 1.0
                    duration = cached_segments[-1]['end'] if cached_segments else 0.0
                info = MockInfo()
            elif self.use_mlx:
                # === MLX WHISPER + SILERO VAD PATH ===
                print("Whisper Thread: Using MLX Whisper + Silero VAD pipeline")

                # Step 1: Run Silero VAD to detect speech segments
                self.emit_progress(20, "Running Silero VAD (speech detection)...")
                speech_segments = run_silero_vad(self.audio_path, self.vad_parameters)

                if self._is_cancelled: raise InterruptedError("Processing cancelled")

                if not speech_segments:
                    print("MLX Whisper: No speech detected by Silero VAD")
                    faster_whisper_segments = []
                    info = MockInfo()
                else:
                    # Step 2: Transcribe speech segments with MLX Whisper
                    self.emit_progress(40, f"Transcribing with MLX Whisper...")
                    faster_whisper_segments = transcribe_with_mlx(
                        self.audio_path,
                        self.model_size_or_path,
                        speech_segments,
                        language="en",
                        beam_size=5  # Use beam search for best quality
                    )

                    if self._is_cancelled: raise InterruptedError("Processing cancelled")

                    # Create info object for compatibility
                    class MockInfo:
                        language = "en"
                        language_probability = 1.0
                        duration = faster_whisper_segments[-1]['end'] if faster_whisper_segments else 0.0
                    info = MockInfo()

                    # Save to cache
                    if faster_whisper_segments:
                        save_vad_cache(self.audio_path, self.vad_parameters, faster_whisper_segments)
            else:
                # === FASTER-WHISPER TRANSCRIPTION ===
                # Only load model if not already pre-loaded
                if self.whisper_model is None:
                    self.emit_progress(15, f"Loading FasterWhisper model ({self.model_size_or_path})...")

                    # Build model parameters with optional CPU/GPU optimization
                    model_kwargs = {
                        "device": device,
                        "compute_type": compute_type
                    }

                    # Check which parameters are supported
                    import inspect
                    supported_params = inspect.signature(WhisperModel.__init__).parameters

                    # Add CPU parallelization parameters if on CPU and supported
                    if device == "cpu":
                        if 'num_workers' in supported_params and 'cpu_threads' in supported_params:
                            model_kwargs["num_workers"] = num_workers
                            model_kwargs["cpu_threads"] = cpu_threads
                            print(f"FasterWhisper: CPU parallelization enabled - {num_workers} workers, {cpu_threads} threads")
                        else:
                            print("FasterWhisper: CPU parallelization not supported in this version")
                    # Add Flash Attention for CUDA if available (20-30% speedup on compatible GPUs)
                    elif device == "cuda":
                        if 'flash_attention' in supported_params:
                            model_kwargs["flash_attention"] = True
                            print("FasterWhisper: Flash Attention enabled (20-30% speedup on compatible GPUs)")

                    self.whisper_model = WhisperModel(self.model_size_or_path, **model_kwargs)
                    # print(f"FasterWhisper Thread: Model loaded. Type: {type(self.whisper_model)}") # Less verbose
                else:
                    self.emit_progress(15, f"Using pre-loaded FasterWhisper model...")
                    print("FasterWhisper Thread: Using pre-loaded model")

                if self._is_cancelled: raise InterruptedError("Processing cancelled")

                self.emit_progress(25, "Starting transcription (with VAD)...")
                print(f"FasterWhisper Thread: Transcribing {self.audio_path}...")

                # *** Use the stored VAD parameters ***
                print(f"FasterWhisper Thread: Using VAD Filter: True, VAD Parameters: {self.vad_parameters}")

                # Build transcription parameters with optional batch_size
                transcribe_kwargs = {
                    "beam_size": 5,
                    "language": "en",
                    "vad_filter": True,
                    "vad_parameters": self.vad_parameters
                }

                # Add batch_size if supported (1.3-1.5x speedup on newer faster-whisper versions)
                try:
                    import inspect
                    if 'batch_size' in inspect.signature(self.whisper_model.transcribe).parameters:
                        transcribe_kwargs["batch_size"] = 16
                        print("FasterWhisper Thread: Batch processing enabled (batch_size=16)")
                except Exception:
                    pass  # Silently skip if not supported

                segments_iterator, info = self.whisper_model.transcribe(
                    self.audio_path,
                    **transcribe_kwargs
                )
                # *** End VAD Parameter Usage ***

                print(f"FasterWhisper Thread: Detected language '{info.language}' with probability {info.language_probability:.2f}")
                print(f"FasterWhisper Thread: Transcription duration: {info.duration}s")

                faster_whisper_segments = []
                segment_count = 0
                for segment in segments_iterator:
                    if self._is_cancelled: raise InterruptedError("Processing cancelled during transcription iteration")
                    if segment.start is None or segment.end is None or segment.end < segment.start:
                         print(f"FasterWhisper Thread WARN: Skipping invalid segment from iterator: start={segment.start}, end={segment.end}")
                         continue
                    segment_dict = { "start": segment.start, "end": segment.end, "text": segment.text }
                    faster_whisper_segments.append(segment_dict)
                    segment_count += 1

                print(f"FasterWhisper Thread: Transcription complete. Found {segment_count} segments (post-VAD).")

                # === Save to VAD cache ===
                if faster_whisper_segments:
                    save_vad_cache(self.audio_path, self.vad_parameters, faster_whisper_segments)
                    print(f"VAD Cache: Saved {len(faster_whisper_segments)} segments to cache")

            # --- START DEBUG LOG ---
            print(f"--- FasterWhisper Raw Segments (Before Alignment) ---")
            if faster_whisper_segments:
                for idx, seg_raw in enumerate(faster_whisper_segments):
                    print(f"  Raw Seg {idx}: Start={seg_raw.get('start', '?'):.3f} End={seg_raw.get('end', '?'):.3f} Text=\"{seg_raw.get('text', '')}\"")
            else:
                print("  (No raw segments found by FasterWhisper VAD)")
            print(f"--- End Raw Segments Log ---")
            # --- END DEBUG LOG ---

            if self._is_cancelled: raise InterruptedError("Processing cancelled")

            # Memory Cleanup for ASR model
            try:
                if hasattr(self, 'whisper_model') and self.whisper_model:
                    del self.whisper_model; self.whisper_model = None; print("FasterWhisper Thread: Whisper ASR model released.")
                if TORCH_AVAILABLE and device == 'cuda': import gc; gc.collect(); torch.cuda.empty_cache()
            except Exception: pass

            # Alignment Step (using WhisperX) - optional for 30-40% speedup when disabled
            if WHISPERX_ALIGN_AVAILABLE and faster_whisper_segments and not self.skip_alignment:
                self.emit_progress(80, "Loading alignment model...")
                try:
                    if hasattr(self, 'align_model') and self.align_model:
                        del self.align_model; del self.align_metadata;
                        self.align_model = None; self.align_metadata = None
                        if TORCH_AVAILABLE and device == 'cuda': import gc; gc.collect(); torch.cuda.empty_cache()

                    self.align_model, self.align_metadata = whisperx.load_align_model(language_code=info.language, device=device)
                    print("WhisperX Thread: Alignment model loaded.")
                    if self._is_cancelled: raise InterruptedError("Processing cancelled")

                    self.emit_progress(85, "Performing alignment...")
                    print(f"WhisperX Thread: Aligning {len(faster_whisper_segments)} segments...")
                    result_aligned = whisperx.align(
                        faster_whisper_segments, self.align_model, self.align_metadata,
                        self.audio_path, device, return_char_alignments=False
                    )
                    print("WhisperX Thread: Alignment complete.")
                    all_segments = result_aligned.get("segments", [])
                    if not all_segments and faster_whisper_segments:
                        print("WhisperX Warning: Alignment resulted in empty segments list. Falling back to pre-alignment segments.")
                        all_segments = faster_whisper_segments
                        for seg in all_segments: seg['words'] = []
                except Exception as e_align:
                    error_msg = f"WhisperX alignment failed: {type(e_align).__name__}: {e_align}. Using transcription segments without word alignment."
                    print(f"WhisperX Warning: {error_msg}")
                    all_segments = faster_whisper_segments
                    for seg in all_segments: seg['words'] = []
            else:
                 if self.skip_alignment:
                     print("WhisperX Thread: Skipping alignment (disabled for faster processing)")
                 elif not faster_whisper_segments:
                     print("WhisperX Thread: No segments found by VAD/Transcription.")
                 else:
                     print("WhisperX Thread: Skipping alignment (WhisperX library not available or no segments).")
                 all_segments = faster_whisper_segments
                 for seg in all_segments: seg['words'] = []

            if self._is_cancelled: raise InterruptedError("Processing cancelled")

            # Process Final Results
            self.emit_progress(95, "Processing results...")
            if all_segments:
                print(f"FasterWhisper Thread: Found {len(all_segments)} final segments.")
                for i, seg in enumerate(all_segments):
                    seg['id'] = i
                    if 'words' not in seg: seg['words'] = []

                # --- COMMENTED OUT Detailed Log ---
                # print("--- Final Word Timestamps Log ---")
                # for i, segment in enumerate(all_segments):
                #     print(f"  Segment {segment.get('id', i)} ({segment.get('start', '?'):.3f}s - {segment.get('end', '?'):.3f}s): \"{segment.get('text', '')}\"")
                #     words = segment.get('words', [])
                #     if words:
                #         for word_info in words:
                #             word = word_info.get('word', ''); start = word_info.get('start', None); end = word_info.get('end', None); score = word_info.get('score', word_info.get('probability', None))
                #             start_str = f"{start:>7.3f}s" if start is not None else "  N/A   "; end_str = f"{end:>7.3f}s" if end is not None else "  N/A   "; score_str = f"{score:.3f}" if score is not None else " N/A "
                #             print(f"    - {word:<15} Start: {start_str}, End: {end_str}, Score: {score_str}")
                #     else: print(f"    - (No word timestamps in this segment)")
                # print("--- End Log ---")
                # --- End COMMENTED OUT Detailed Log ---
            else:
                print(f"FasterWhisper Thread: No segments in final result.")

            # Emit final results
            if not self._is_cancelled:
                self.emit_progress(100, "Processing finished.")
                print(f"FasterWhisper Thread: Emitting final results ({len(all_segments)} segments).")
                self.processing_finished.emit(all_segments)

        except InterruptedError:
            print(f"FasterWhisper Thread: Processing cancelled.")
            self.processing_finished.emit([])
        except FileNotFoundError as e:
             print(f"FasterWhisper Thread Error: File not found - {e}")
             self.processing_error.emit(f"File not found: {e}")
             self.processing_finished.emit([])
        except Exception as e:
            error_msg = f"Unexpected error in FasterWhisper thread: {type(e).__name__}: {e}"
            print(f"FasterWhisper Thread Error: {error_msg}")
            import traceback; traceback.print_exc()
            self.processing_error.emit(error_msg)
            self.processing_finished.emit([])
        finally:
            # Model Cleanup
            if hasattr(self, 'align_model') and self.align_model:
                 try: del self.align_model; del self.align_metadata; self.align_model = None; self.align_metadata = None; print("FasterWhisper Thread: Alignment model released.")
                 except Exception: pass
            if hasattr(self, 'whisper_model') and self.whisper_model:
                 try: del self.whisper_model; self.whisper_model = None; print("FasterWhisper Thread: Whisper ASR model released (in finally).")
                 except Exception: pass
            if TORCH_AVAILABLE and device == 'cuda':
                 try: import gc; gc.collect(); torch.cuda.empty_cache(); print("FasterWhisper Thread: Final GPU Memory Cleared.")
                 except Exception: pass
            # File cleanup handled externally
            # print(f"FasterWhisper Thread [{self.currentThreadId()}]: Run method finished.") # Less verbose

    # --- Cleanup methods remain the same ---
    def cleanup(self):
        # print(f"FasterWhisper Thread [{self.currentThreadId()}]: External cleanup called.") # Less verbose
        self._cleanup_files_only()

    def _cleanup_files_only(self):
         audio_path_to_clean = self.audio_path; dir_to_clean = self.temp_dir_for_cleanup
         # print(f"FasterWhisper Thread: Cleanup. Audio: {audio_path_to_clean}, Dir: {dir_to_clean}") # Less verbose
         if not self.debug_keep_audio:
            if audio_path_to_clean and os.path.exists(audio_path_to_clean):
                 try: os.remove(audio_path_to_clean); # print(f"FasterWhisper Thread: Removed temp audio: {audio_path_to_clean}") # Less verbose
                 except Exception as e: print(f"FasterWhisper Warning: Failed to remove temp audio file {audio_path_to_clean}: {e}")
            if dir_to_clean and os.path.isdir(dir_to_clean) and "actioncoder_audio_" in os.path.basename(dir_to_clean):
                try: shutil.rmtree(dir_to_clean); # print(f"FasterWhisper Thread: Removed temp dir: {dir_to_clean}") # Less verbose
                except Exception as e: print(f"FasterWhisper Warning: Failed to remove temp directory {dir_to_clean}: {e}")
            # else: # Less verbose
            #     if dir_to_clean: print(f"FasterWhisper Thread: Did not remove directory {dir_to_clean} (exists={os.path.isdir(dir_to_clean)}, name_match={'actioncoder_audio_' in os.path.basename(dir_to_clean)}).")
         else: print(f"FasterWhisper Thread: Skipping cleanup (debug_keep_audio=True).")
         self.audio_path = None; self.temp_dir_for_cleanup = None

    def emit_progress(self, percentage, message):
        percentage = max(0, min(100, percentage)); self.progress_update.emit(percentage, message)

    def stop(self):
        print(f"FasterWhisper Thread [{self.currentThreadId()}]: Received stop signal.")
        self._is_cancelled = True

