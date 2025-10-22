# processing_manager.py

import os
import subprocess
import shutil
import tempfile
import sys
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread

# --- NEW IMPORTS for Strategy 2 ---
import soundfile as sf
import numpy as np
import struct
import webrtcvad
from collections import Counter # For potential median calculation or statistics
# --- END NEW IMPORTS ---

from whisper_handler import WhisperHandler
from video_processor import VideoProcessor # Import VideoProcessor
import traceback # For error logging

# --- ProcessingThread Definition (Unchanged) ---
class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int); finished_signal = pyqtSignal(); error_signal = pyqtSignal(str)

    def __init__(self, video_path, action_tracker, csv_handler, action_dict, output_csv, output_video, output_csv2=None, parent=None):
        super().__init__(parent);
        self.video_path=video_path
        self.action_tracker=action_tracker # Keep reference for get_action_for_frame
        self.csv_handler=csv_handler
        self.action_dict=action_dict
        self.output_csv=output_csv
        self.output_video=output_video
        self.output_csv2=output_csv2

    def run(self):
        try:
            print("Processing Thread: Starting..."); self.progress_signal.emit(10)

            if not self.action_dict:
                print("Processing Thread: Warning - No actions were recorded (empty dict passed).")

            print("Processing Thread: Adding action column to CSV(s)...")
            # Ensure action_dict is passed to add_action_column
            if not self.csv_handler.add_action_column(self.action_dict):
                 print("Processing Thread: Warning - Failed to add action column fully to one or both CSVs.")
            self.progress_signal.emit(30)

            print(f"Processing Thread: Saving CSV(s) to {self.output_csv} and {self.output_csv2}...")
            if not self.csv_handler.save_data(self.output_csv, self.output_csv2):
                 print("Processing Thread: Warning - Failed to save one or both CSV files.")
            self.progress_signal.emit(50)

            # --- Initialize VideoProcessor HERE ---
            print("Processing Thread: Initializing video processor for output video...")
            video_processor = VideoProcessor(self.video_path, self.action_tracker)
            # init_video() is called inside video_processor.process_video()
            # --- End Initialization ---

            def report_video_progress(progress):
                # Calculate progress for the video part (50% to 100%)
                self.progress_signal.emit(50 + int(progress * 0.5))

            print(f"Processing Thread: Processing video for overlays -> {self.output_video}...")
            # Pass action_dict to process_video if needed, otherwise action_tracker is used internally
            if not video_processor.process_video(self.output_video, report_video_progress):
                 # If video processing fails, emit an error but still finish "successfully"
                 # as the CSVs might be okay. The controller already warned about placeholders.
                 error_msg = f"Video processing failed for {os.path.basename(self.output_video)}. Check logs."
                 print(f"Processing Thread WARNING: {error_msg}")
                 # Optionally emit a specific non-critical error signal? For now, just log.
            else:
                print("Processing Thread: Video processing finished.")

            self.progress_signal.emit(100);
            self.finished_signal.emit();
            print("Processing Thread: Finished.")
        except Exception as e:
            error_msg = f"Processing error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"Processing Thread Error: {error_msg}");
            self.error_signal.emit(error_msg) # Emit critical error

# --- End ProcessingThread Definition ---


class ProcessingManager(QObject):
    """Manages background processing tasks (Whisper, Saving Output)."""

    # Signals to Controller/UIManager
    processing_status_update = pyqtSignal(str) # General status messages
    processing_progress_update = pyqtSignal(int) # For save progress bar
    whisper_results_ready = pyqtSignal(list) # Emits Whisper segments
    save_complete = pyqtSignal(bool) # True if successful (even with warnings), False if critical error occurred
    processing_error = pyqtSignal(str, str) # Type ("whisper", "save", "audio"), Message

    def __init__(self, timeline_processor, ffmpeg_path, parent=None, whisper_model=None):
        super().__init__(parent)
        self.timeline_processor = timeline_processor
        self.ffmpeg_path = ffmpeg_path
        self.whisper_thread = None
        self.save_thread = None
        self.current_whisper_handler = None
        self.current_temp_audio_dir = None
        self.previous_whisper_handler = None # Keep track for cleanup
        self.preloaded_whisper_model = whisper_model # Store pre-loaded Whisper model

    def cleanup_previous_whisper_files(self):
        """Cleans up temp files from the *previous* Whisper run."""
        if self.previous_whisper_handler:
            print(f"ProcessingManager: Cleaning up previous Whisper handler files: {getattr(self.previous_whisper_handler, 'temp_dir_for_cleanup', 'N/A')}")
            self.previous_whisper_handler.cleanup()
            self.previous_whisper_handler = None

    def _calculate_rms_dbfs(self, audio_path):
        """Calculates the average RMS level of an audio file in dBFS."""
        try:
            data, samplerate = sf.read(audio_path)
            if samplerate not in [8000, 16000, 32000, 48000]:
                 print(f"ProcessingManager WARN: Unsupported sample rate {samplerate} for webrtcvad. Cannot perform preliminary VAD.")
                 return -999.0, None
            if data.ndim > 1: data = np.mean(data, axis=1)
            if len(data) == 0: return -999.0, None
            rms = np.sqrt(np.mean(data**2))
            dbfs = -999.0 if rms == 0 else 20 * np.log10(rms)
            return dbfs, samplerate
        except Exception as e:
            print(f"ProcessingManager ERROR: Failed to calculate RMS for {audio_path}: {e}")
            return -999.0, None

    def _run_preliminary_vad(self, audio_path, sample_rate):
        """Runs webrtcvad to get speech/silence duration statistics."""
        if sample_rate not in [8000, 16000, 32000, 48000]: return None
        speech_durations_ms = []; silence_durations_ms = []
        try:
            vad = webrtcvad.Vad(); vad.set_mode(3)
            frame_duration_ms = 30; frame_samples = int(sample_rate * frame_duration_ms / 1000); bytes_per_sample = 2
            current_state_is_speech = None; current_run_length = 0
            with sf.SoundFile(audio_path, 'r') as f:
                if f.channels > 1: print("ProcessingManager WARN: Preliminary VAD processing stereo file - will process first channel only.")
                for block in f.blocks(blocksize=frame_samples, dtype='int16', fill_value=0):
                    if block.shape[0] == 0: break
                    mono_block = block[:, 0] if block.ndim > 1 else block
                    if len(mono_block) < frame_samples: mono_block = np.concatenate((mono_block, np.zeros(frame_samples - len(mono_block), dtype='int16')))
                    try: byte_data = struct.pack(f'{frame_samples}h', *mono_block)
                    except struct.error as e: print(f"ProcessingManager ERROR: struct.pack failed: {e}. Skipping."); continue
                    try: is_speech = vad.is_speech(byte_data, sample_rate)
                    except Exception as vad_err: print(f"ProcessingManager WARN: webrtcvad.is_speech failed: {vad_err}. Skipping."); continue
                    if current_state_is_speech is None: current_state_is_speech = is_speech; current_run_length = 1
                    elif is_speech == current_state_is_speech: current_run_length += 1
                    else:
                        duration_ms = current_run_length * frame_duration_ms
                        if current_state_is_speech: speech_durations_ms.append(duration_ms)
                        else: silence_durations_ms.append(duration_ms)
                        current_state_is_speech = is_speech; current_run_length = 1
            if current_run_length > 0:
                duration_ms = current_run_length * frame_duration_ms
                if current_state_is_speech: speech_durations_ms.append(duration_ms)
                else: silence_durations_ms.append(duration_ms)
            median_speech_ms = np.median(speech_durations_ms) if speech_durations_ms else 0
            median_silence_ms = np.median(silence_durations_ms) if silence_durations_ms else 0
            return {"median_speech_ms": median_speech_ms, "median_silence_ms": median_silence_ms,
                    "num_speech_segments": len(speech_durations_ms), "num_silence_segments": len(silence_durations_ms)}
        except ImportError: print("ProcessingManager ERROR: webrtcvad library not found."); return None
        except Exception as e: print(f"ProcessingManager ERROR: Failed during preliminary VAD analysis: {e}"); return None


    def _get_dynamic_vad_params(self, rms_dbfs, vad_stats):
        """Determines VAD parameters based on loudness and preliminary VAD statistics."""

        default_threshold = 0.5; default_min_silence_ms = 250; default_min_speech_ms = 200; default_padding_ms = 300
        MIN_THRESHOLD = 0.35; MAX_THRESHOLD = 0.6; MIN_SILENCE_MS = 100; MAX_SILENCE_MS = 700
        MIN_SPEECH_MS = 100; MAX_SPEECH_MS = 500; MIN_PADDING_MS = 150; MAX_PADDING_MS = 400
        VERY_QUIET_THRESHOLD_DBFS = -35.0; QUIET_THRESHOLD_DBFS = -25.0

        # --- Threshold ---
        if rms_dbfs == -999.0: vad_threshold = default_threshold; print("  DynamicVAD: Using default threshold (RMS error/silence).")
        elif rms_dbfs < VERY_QUIET_THRESHOLD_DBFS: vad_threshold = 0.40; print(f"  DynamicVAD: Loudness ({rms_dbfs:.1f} dBFS) -> Threshold: {vad_threshold:.2f} (Very Quiet)")
        elif rms_dbfs < QUIET_THRESHOLD_DBFS: vad_threshold = 0.45; print(f"  DynamicVAD: Loudness ({rms_dbfs:.1f} dBFS) -> Threshold: {vad_threshold:.2f} (Quiet)")
        else: vad_threshold = default_threshold; print(f"  DynamicVAD: Loudness ({rms_dbfs:.1f} dBFS) -> Threshold: {vad_threshold:.2f} (Default/Loud)")
        vad_threshold = max(MIN_THRESHOLD, min(vad_threshold, MAX_THRESHOLD))

        # --- Timing ---
        min_silence_ms = default_min_silence_ms; min_speech_ms = default_min_speech_ms; padding_ms = default_padding_ms
        if vad_stats and vad_stats["median_silence_ms"] > 0 and vad_stats["median_speech_ms"] > 0:
            median_silence = vad_stats["median_silence_ms"]; median_speech = vad_stats["median_speech_ms"]
            print(f"  DynamicVAD: Prelim Stats (Silence: {median_silence:.0f}ms, Speech: {median_speech:.0f}ms)")
            min_silence_ms = int(median_silence * 0.6)
            min_speech_ms = int(median_speech * 0.7)

            # *** VAD TWEAK: Add a ceiling to prevent min_speech_duration being too high ***
            MAX_CALCULATED_MIN_SPEECH = 180 # Ceiling in ms (tune this value if needed)
            if min_speech_ms > MAX_CALCULATED_MIN_SPEECH:
                print(f"    -> Clamping calculated min_speech {min_speech_ms}ms down to {MAX_CALCULATED_MIN_SPEECH}ms ceiling")
                min_speech_ms = MAX_CALCULATED_MIN_SPEECH
            # *** END VAD TWEAK ***

        else:
            print("  DynamicVAD: Using default timing params (prelim VAD failed or no speech/silence found).")

        # --- Clamp all ---
        final_min_silence_ms = max(MIN_SILENCE_MS, min(min_silence_ms, MAX_SILENCE_MS))
        final_min_speech_ms = max(MIN_SPEECH_MS, min(min_speech_ms, MAX_SPEECH_MS)) # Apply final bounds
        final_padding_ms = max(MIN_PADDING_MS, min(padding_ms, MAX_PADDING_MS))

        vad_params = {
            "threshold": vad_threshold,
            "min_speech_duration_ms": final_min_speech_ms, # Use final clamped value
            "max_speech_duration_s": float("inf"),
            "min_silence_duration_ms": final_min_silence_ms,
            "speech_pad_ms": final_padding_ms
        }
        print(f"  DynamicVAD: Final Params -> {vad_params}")
        return vad_params

    @pyqtSlot(str)
    def start_whisper_processing(self, video_path):
        # (No changes needed in the rest of the method)
        if not video_path or not os.path.exists(video_path):
            self.processing_error.emit("whisper", f"Invalid video path: {video_path}")
            return
        if self.whisper_thread and self.whisper_thread.isRunning():
            print("ProcessingManager: Whisper processing already running.")
            self.processing_status_update.emit("Whisper already running")
            return
        self.cleanup_previous_whisper_files()
        self.previous_whisper_handler = self.current_whisper_handler
        self.current_whisper_handler = None; self.current_temp_audio_dir = None
        print(f"ProcessingManager: Starting Whisper process for: {video_path}")
        self.processing_status_update.emit("Extracting audio...")
        if not self.ffmpeg_path:
            self.processing_error.emit("audio", "ffmpeg not found. Cannot extract audio.")
            self.processing_status_update.emit("Error: ffmpeg missing"); return
        temp_dir = None; temp_audio_path = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="actioncoder_audio_"); temp_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
            self.current_temp_audio_dir = temp_dir
        except Exception as e:
            error_msg = f"Failed to create temporary directory for audio: {e}"
            print(f"ProcessingManager ERROR: {error_msg}"); self.processing_error.emit("audio", error_msg); self.processing_status_update.emit(f"Error: {error_msg}")
            if temp_dir and os.path.isdir(temp_dir): self._cleanup_temp_dir(temp_dir); self.current_temp_audio_dir = None; return
        success, error_msg = self._extract_audio_direct(video_path, temp_audio_path)
        if not success:
            full_error = f"Failed to extract audio using direct ffmpeg call. Error: {error_msg}"
            print(f"ProcessingManager ERROR: {full_error}"); self.processing_error.emit("audio", full_error); self.processing_status_update.emit("Error: Audio extraction failed")
            if self.current_temp_audio_dir and os.path.isdir(self.current_temp_audio_dir): self._cleanup_temp_dir(self.current_temp_audio_dir); self.current_temp_audio_dir = None; return
        self.processing_status_update.emit("Analyzing audio characteristics...")
        rms_dbfs, sample_rate = self._calculate_rms_dbfs(temp_audio_path)
        vad_stats = self._run_preliminary_vad(temp_audio_path, sample_rate)
        dynamic_vad_params = self._get_dynamic_vad_params(rms_dbfs, vad_stats) # <--- VAD tweak happens inside here
        print(f"ProcessingManager: Using dynamically determined VAD Params: {dynamic_vad_params}")
        self.processing_status_update.emit("Audio analyzed. Starting Whisper...")
        try:
            self.whisper_thread = WhisperHandler(audio_path=temp_audio_path, temp_dir_for_cleanup=self.current_temp_audio_dir, vad_parameters=dynamic_vad_params, model_name="large-v3", debug_keep_audio=False, preloaded_model=self.preloaded_whisper_model)
            self.current_whisper_handler = self.whisper_thread
            self.whisper_thread.progress_update.connect(self._handle_whisper_progress)
            self.whisper_thread.processing_error.connect(self._handle_whisper_error)
            self.whisper_thread.processing_finished.connect(self._handle_whisper_finished)
            self.whisper_thread.finished.connect(self._on_whisper_thread_finished)
            self.whisper_thread.start()
        except Exception as e_thread:
            error_msg = f"Failed to initialize or start WhisperHandler thread: {e_thread}"
            print(f"ProcessingManager ERROR: {error_msg}"); self.processing_error.emit("whisper", error_msg); self.processing_status_update.emit(f"Error: {error_msg}")
            if self.current_temp_audio_dir and os.path.isdir(self.current_temp_audio_dir): self._cleanup_temp_dir(self.current_temp_audio_dir); self.current_temp_audio_dir = None; self.current_whisper_handler = None


    @pyqtSlot()
    def stop_whisper_processing(self, wait=False):
        # (No changes)
        thread_to_stop = self.whisper_thread
        if thread_to_stop and thread_to_stop.isRunning():
            print(f"ProcessingManager: Stopping WhisperX thread {thread_to_stop}...");
            thread_to_stop.stop()
            if wait:
                finished = thread_to_stop.wait(3000)
                print(f"ProcessingManager: Whisper thread wait finished: {finished}")
            self.processing_status_update.emit("Idle (Stopped)")

    @pyqtSlot(str, 'QObject', 'QObject', dict, str, str, object)
    def start_save_processing(self, video_path, action_tracker, csv_handler, action_dict, out_csv, out_vid, out_csv2=None):
        # (No changes)
        if self.save_thread and self.save_thread.isRunning():
            print("ProcessingManager: Save processing already running.")
            return
        print(f"ProcessingManager: Starting save process for {os.path.basename(video_path)} -> {os.path.dirname(out_csv)}")
        self.processing_progress_update.emit(0)
        self.save_thread = ProcessingThread(video_path, action_tracker, csv_handler, action_dict, out_csv, out_vid, out_csv2)
        self.save_thread.progress_signal.connect(self.processing_progress_update)
        self.save_thread.finished_signal.connect(self._handle_save_finished)
        self.save_thread.error_signal.connect(self._handle_save_error)
        self.save_thread.finished.connect(self._on_save_thread_finished)
        self.save_thread.start()

    # --- Internal Slots for Thread Signals ---
    @pyqtSlot(int, str)
    def _handle_whisper_progress(self, percentage, message): # (No change)
        self.processing_status_update.emit(f"{message} ({percentage}%)")
    @pyqtSlot(str)
    def _handle_whisper_error(self, error_message): # (No change)
        print(f"ProcessingManager ERROR: Whisper processing failed: {error_message}")
        self.processing_error.emit("whisper", error_message)
        self.processing_status_update.emit(f"Error: {error_message}")
    @pyqtSlot(list)
    def _handle_whisper_finished(self, segments): # (No change)
        print(f"ProcessingManager: Whisper finished successfully ({len(segments)} segments).")
        self.whisper_results_ready.emit(segments)
        self.processing_status_update.emit("Ready.")
    @pyqtSlot()
    def _on_whisper_thread_finished(self): # (No change)
        print("ProcessingManager: Whisper thread finished.")
        self.whisper_thread = None
    @pyqtSlot()
    def _handle_save_finished(self): # (No change)
        print("ProcessingManager: Save process finished.")
        self.processing_progress_update.emit(100)
        self.save_complete.emit(True)
    @pyqtSlot(str)
    def _handle_save_error(self, error_message): # (No change)
        print(f"ProcessingManager ERROR: Save process failed critically: {error_message}")
        self.processing_error.emit("save", error_message)
        self.save_complete.emit(False)
    @pyqtSlot()
    def _on_save_thread_finished(self): # (No change)
        print("ProcessingManager: Save thread finished.")
        self.save_thread = None

    # --- Helper Methods ---
    def _extract_audio_direct(self, video_path, output_wav_path):
        # (No changes here - noise reduction stays commented out unless testing)
        if not self.ffmpeg_path: return False, "ffmpeg path not found."
        command = [ self.ffmpeg_path, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_wav_path ]
        try:
            startupinfo = None; creationflags = 0
            if sys.platform == 'win32': startupinfo = subprocess.STARTUPINFO(); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startupinfo.wShowWindow = subprocess.SW_HIDE; creationflags=subprocess.CREATE_NO_WINDOW
            result = subprocess.run(command, check=False, capture_output=True, text=True, startupinfo=startupinfo, creationflags=creationflags)
            if result.returncode != 0:
                error_message = f"ffmpeg failed with code {result.returncode}.\nStderr:\n{result.stderr}"
                print(f"ProcessingManager ERROR: {error_message}"); self._cleanup_temp_file(output_wav_path); return False, error_message
            else:
                try:
                    with sf.SoundFile(output_wav_path) as f_check:
                        if f_check.samplerate != 16000: error_message = f"ffmpeg succeeded but output sample rate is {f_check.samplerate}, not 16000Hz!"; print(f"ProcessingManager ERROR: {error_message}"); self._cleanup_temp_file(output_wav_path); return False, error_message
                        if f_check.channels != 1: error_message = f"ffmpeg succeeded but output channels is {f_check.channels}, not 1 (Mono)!"; print(f"ProcessingManager ERROR: {error_message}"); self._cleanup_temp_file(output_wav_path); return False, error_message
                        if f_check.subtype != 'PCM_16': error_message = f"ffmpeg succeeded but output format is {f_check.subtype}, not PCM_16!"; print(f"ProcessingManager ERROR: {error_message}"); self._cleanup_temp_file(output_wav_path); return False, error_message
                except Exception as sf_err: error_message = f"Error verifying extracted audio format with soundfile: {sf_err}"; print(f"ProcessingManager ERROR: {error_message}"); self._cleanup_temp_file(output_wav_path); return False, error_message
                return True, None
        except FileNotFoundError: error_message = f"ffmpeg command not found at '{self.ffmpeg_path}'."; print(f"ProcessingManager ERROR: {error_message}"); return False, error_message
        except Exception as e: error_message = f"Unexpected error during direct ffmpeg call: {type(e).__name__}: {e}\n{traceback.format_exc()}"; print(f"ProcessingManager ERROR: {error_message}"); self._cleanup_temp_file(output_wav_path); return False, error_message

    def _cleanup_temp_dir(self, dir_path): # (No change)
        if dir_path and os.path.isdir(dir_path):
            try: shutil.rmtree(dir_path)
            except Exception as e: print(f"ProcessingManager WARN: Failed to remove temp directory {dir_path}: {e}")

    def _cleanup_temp_file(self, file_path): # (No change)
        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except Exception as e: print(f"ProcessingManager WARN: Failed to remove temp file {file_path}: {e}")

    def cleanup_on_exit(self): # (No change)
        print("ProcessingManager: Cleanup on exit..."); self.stop_whisper_processing(wait=True); self.cleanup_previous_whisper_files()
        if self.current_whisper_handler: print(f"ProcessingManager: Cleaning up current/last Whisper handler files: {self.current_temp_audio_dir}"); self.current_whisper_handler.cleanup(); self.current_whisper_handler = None; self.current_temp_audio_dir = None
        elif self.current_temp_audio_dir: print(f"ProcessingManager: Cleaning up orphaned temp audio directory: {self.current_temp_audio_dir}"); self._cleanup_temp_dir(self.current_temp_audio_dir); self.current_temp_audio_dir = None
        if self.save_thread and self.save_thread.isRunning(): print("ProcessingManager: Waiting for save thread on exit..."); self.save_thread.wait()

