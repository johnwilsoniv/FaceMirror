# --- START OF FILE whisper_handler.py ---

# whisper_handler.py - Handles audio extraction and Whisper transcription/alignment in a separate thread
import os
import sys
import tempfile
import shutil
import time # For potential sleep/yield

# --- Use faster-whisper directly ---
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("Whisper Handler: FasterWhisper library found.")
except ImportError as e:
    print(f"Whisper Handler ERROR: FasterWhisper import failed: {e}. Cannot perform transcription.")
    WhisperModel = None # Define as None if import fails
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
import builtins # Import builtins to check for InterruptedError

# --- Try importing torch ---
try:
    import torch
    TORCH_AVAILABLE = True
    print("Whisper Handler: PyTorch found.")
except ImportError:
    TORCH_AVAILABLE = False
    print("Whisper Handler: PyTorch not found, will use CPU.")
# --- End torch import ---


# Define InterruptedError if it doesn't exist
if 'InterruptedError' not in builtins.__dict__:
    class InterruptedError(Exception):
        pass

class WhisperHandler(QThread):
    """
    Runs FasterWhisper transcription + WhisperX alignment in a separate thread.
    Uses FasterWhisper's VAD parameters (potentially dynamic) during transcription.
    Processes a pre-extracted audio file.
    Returns segments list (with words) on finish.
    """
    progress_update = pyqtSignal(int, str)
    processing_error = pyqtSignal(str)
    processing_finished = pyqtSignal(list) # Emits results list (aligned segments with words)

    audio_path = None
    temp_dir_for_cleanup = None

    # --- MODIFIED __init__ to accept vad_parameters ---
    def __init__(self, audio_path, temp_dir_for_cleanup, vad_parameters, model_name="large-v3", parent=None, debug_keep_audio=False):
        super().__init__(parent)
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path does not exist or not provided: {audio_path}")
        if not FASTER_WHISPER_AVAILABLE:
             raise ImportError("FasterWhisper library is required but not installed.")
        if not WHISPERX_ALIGN_AVAILABLE:
            print("WhisperHandler WARNING: WhisperX not available, alignment will be skipped.")


        self.audio_path = audio_path
        self.temp_dir_for_cleanup = temp_dir_for_cleanup
        self.model_size_or_path = model_name
        self.vad_parameters = vad_parameters # Store the passed VAD parameters
        self._is_cancelled = False
        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None
        self.debug_keep_audio = debug_keep_audio
        # print(f"WhisperHandler Initialized with audio: {self.audio_path}, temp_dir: {self.temp_dir_for_cleanup}") # Less verbose
        print(f"WhisperHandler Using VAD Parameters: {self.vad_parameters}") # Log params used

    # --- MODIFIED run method to use self.vad_parameters ---
    def run(self):
        self._is_cancelled = False
        all_segments = []
        device = "cpu"
        # print(f"FasterWhisper Thread [{self.currentThreadId()}]: Starting processing for {self.audio_path}") # Less verbose

        if not FASTER_WHISPER_AVAILABLE:
            self.processing_error.emit("FasterWhisper library not installed.")
            self.processing_finished.emit([])
            return

        try:
            if not self.audio_path or not os.path.exists(self.audio_path):
                 raise FileNotFoundError("Pre-extracted audio path not valid during run.")

            # print(f"FasterWhisper Thread: Using pre-extracted audio: {self.audio_path}") # Less verbose

            self.emit_progress(10, f"Setting up device...")
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            if device == "cpu": print("FasterWhisper Warning: Running on CPU.")
            print(f"FasterWhisper Thread: Using device: {device}, compute_type: {compute_type}")

            self.emit_progress(15, f"Loading FasterWhisper model ({self.model_size_or_path})...")
            self.whisper_model = WhisperModel(self.model_size_or_path, device=device, compute_type=compute_type)
            # print(f"FasterWhisper Thread: Model loaded. Type: {type(self.whisper_model)}") # Less verbose
            if self._is_cancelled: raise InterruptedError("Processing cancelled")

            self.emit_progress(25, "Starting transcription (with VAD)...")
            print(f"FasterWhisper Thread: Transcribing {self.audio_path}...")

            # *** Use the stored VAD parameters ***
            print(f"FasterWhisper Thread: Using VAD Filter: True, VAD Parameters: {self.vad_parameters}")

            segments_iterator, info = self.whisper_model.transcribe(
                self.audio_path,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=self.vad_parameters # Use the member variable
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

            # Alignment Step (using WhisperX)
            if WHISPERX_ALIGN_AVAILABLE and faster_whisper_segments:
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
                 if not faster_whisper_segments: print("WhisperX Thread: No segments found by VAD/Transcription.")
                 else: print("WhisperX Thread: Skipping alignment (WhisperX library not available or no segments).");
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

# --- END OF FILE whisper_handler.py ---