# --- START OF FILE analyze_audio.py ---

# analyze_audio_gui.py
import sys
import os
import soundfile as sf
import numpy as np
import statistics # For median calculation

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
    QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt

def analyze_audio(filepath, silence_threshold_dbfs=-30.0, min_meaningful_silence_ms=150): # Added min duration filter
    """Analyzes a WAV file for characteristics relevant to VAD tuning."""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}

    try:
        data, samplerate = sf.read(filepath, dtype='int16')
    except Exception as e:
        return {"error": f"Error reading {filepath}: {e}"}

    if samplerate != 16000:
        print(f"Warning: Sample rate is {samplerate}Hz, expected 16000Hz for {filepath}")

    if data.ndim > 1:
        print(f"Warning: Audio is not mono ({data.shape}), converting for analysis ({filepath})")
        data = data.mean(axis=1).astype(np.int16) # Average channels for mono

    duration_sec = len(data) / samplerate
    peak_amplitude = np.max(np.abs(data))
    max_possible_amplitude = 32767 # for int16
    peak_dbfs = 20 * np.log10(peak_amplitude / max_possible_amplitude) if peak_amplitude > 0 else -np.inf
    clipping_percent = (np.sum(np.abs(data) == max_possible_amplitude) / len(data)) * 100

    rms_amplitude = np.sqrt(np.mean(data.astype(np.float64)**2))
    overall_rms_dbfs = 20 * np.log10(rms_amplitude / max_possible_amplitude) if rms_amplitude > 0 else -np.inf

    # --- Silence Detection (as before, using hardcoded threshold for analysis) ---
    window_size_ms = 50 # Window for RMS calculation
    hop_size_ms = 10    # Step size for window
    window_samples = int(samplerate * window_size_ms / 1000)
    hop_samples = int(samplerate * hop_size_ms / 1000)

    # Convert threshold from dBFS to amplitude
    silence_threshold_amp = max_possible_amplitude * (10**(silence_threshold_dbfs / 20.0))

    num_windows = (len(data) - window_samples) // hop_samples + 1
    is_silent_window = np.zeros(num_windows, dtype=bool)
    window_rms_amps = []

    for i in range(num_windows):
        start = i * hop_samples
        end = start + window_samples
        window = data[start:end].astype(np.float64)
        window_rms = np.sqrt(np.mean(window**2))
        window_rms_amps.append(window_rms)
        if window_rms < silence_threshold_amp:
            is_silent_window[i] = True

    # --- Calculate Segment Durations ---
    silent_segments_raw = []
    current_segment_start = None
    for i in range(num_windows):
        is_silent_now = is_silent_window[i]
        current_time_ms = i * hop_size_ms

        if is_silent_now and current_segment_start is None:
            # Start of a silent segment
            current_segment_start = current_time_ms
        elif not is_silent_now and current_segment_start is not None:
            # End of a silent segment
            end_time_ms = current_time_ms # End is start of the non-silent window
            duration_ms = end_time_ms - current_segment_start
            silent_segments_raw.append({
                'start_ms': current_segment_start,
                'end_ms': end_time_ms,
                'duration_ms': duration_ms
            })
            current_segment_start = None
    # Handle silence at the very end
    if current_segment_start is not None:
        end_time_ms = duration_sec * 1000
        duration_ms = end_time_ms - current_segment_start
        silent_segments_raw.append({
            'start_ms': current_segment_start,
            'end_ms': end_time_ms,
            'duration_ms': duration_ms
        })

    # --- Filter out very short silences & calculate stats ---
    meaningful_silent_segments = [s for s in silent_segments_raw if s['duration_ms'] >= min_meaningful_silence_ms]
    num_meaningful_silent_segments = len(meaningful_silent_segments)
    meaningful_durations_ms = [s['duration_ms'] for s in meaningful_silent_segments]

    min_silence_duration_ms = min(meaningful_durations_ms) if meaningful_durations_ms else 0
    max_silence_duration_ms = max(meaningful_durations_ms) if meaningful_durations_ms else 0
    avg_silence_duration_ms = sum(meaningful_durations_ms) / num_meaningful_silent_segments if num_meaningful_silent_segments > 0 else 0
    median_silence_duration_ms = statistics.median(meaningful_durations_ms) if meaningful_durations_ms else 0

    # --- Calculate RMS levels for detected silence and speech ---
    silent_samples = np.concatenate([
        data[int(s['start_ms']*samplerate/1000):int(s['end_ms']*samplerate/1000)]
        for s in meaningful_silent_segments
    ]) if meaningful_silent_segments else np.array([], dtype=np.int16)

    noise_rms_amplitude = np.sqrt(np.mean(silent_samples.astype(np.float64)**2)) if len(silent_samples) > 0 else 0
    noise_rms_dbfs = 20 * np.log10(noise_rms_amplitude / max_possible_amplitude) if noise_rms_amplitude > 0 else -np.inf

    # Simple way to get speech samples: where the window wasn't silent
    speech_window_indices = np.where(~is_silent_window)[0]
    speech_samples = np.concatenate([
        data[i*hop_samples : i*hop_samples+window_samples]
        for i in speech_window_indices
    ]) if len(speech_window_indices) > 0 else np.array([], dtype=np.int16)

    speech_rms_amplitude = np.sqrt(np.mean(speech_samples.astype(np.float64)**2)) if len(speech_samples) > 0 else 0
    speech_rms_dbfs = 20 * np.log10(speech_rms_amplitude / max_possible_amplitude) if speech_rms_amplitude > 0 else -np.inf

    # Calculate difference
    speech_noise_diff_db = speech_rms_dbfs - noise_rms_dbfs if speech_rms_amplitude > 0 and noise_rms_amplitude > 0 else np.inf

    return {
        "filepath": filepath, "duration_sec": duration_sec, "samplerate": samplerate,
        "peak_dbfs": peak_dbfs, "clipping_percent": clipping_percent,
        "overall_rms_dbfs": overall_rms_dbfs,
        "silence_threshold_dbfs (for analysis)": silence_threshold_dbfs,
        "min_meaningful_silence_ms (filter)": min_meaningful_silence_ms,
        "num_meaningful_silent_segments": num_meaningful_silent_segments,
        "min_silence_duration_ms": min_silence_duration_ms,
        "max_silence_duration_ms": max_silence_duration_ms,
        "avg_silence_duration_ms": avg_silence_duration_ms,
        "median_silence_duration_ms": median_silence_duration_ms,
        "noise_rms_dbfs (in silence)": noise_rms_dbfs,
        "speech_rms_dbfs (in speech)": speech_rms_dbfs,
        "speech_minus_noise_db": speech_noise_diff_db,
        "error": None
    }

# --- GUI Class ---
class AudioAnalyzerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.file1_path = ""
        self.file2_path = ""
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Audio Analyzer for VAD Tuning') # Updated title
        self.setGeometry(200, 200, 750, 650) # Slightly wider

        main_layout = QVBoxLayout()

        # File 1 Selection
        layout1 = QHBoxLayout()
        label1 = QLabel("File 1 (e.g., Good):") # Adjusted label
        self.path1_edit = QLineEdit()
        self.path1_edit.setReadOnly(True)
        browse1_btn = QPushButton("Browse...")
        browse1_btn.clicked.connect(self.browse_file1)
        layout1.addWidget(label1)
        layout1.addWidget(self.path1_edit)
        layout1.addWidget(browse1_btn)
        main_layout.addLayout(layout1)

        # File 2 Selection
        layout2 = QHBoxLayout()
        label2 = QLabel("File 2 (e.g., Bad): ") # Adjusted label
        self.path2_edit = QLineEdit()
        self.path2_edit.setReadOnly(True)
        browse2_btn = QPushButton("Browse...")
        browse2_btn.clicked.connect(self.browse_file2)
        layout2.addWidget(label2)
        layout2.addWidget(self.path2_edit)
        layout2.addWidget(browse2_btn)
        main_layout.addLayout(layout2)

        # Analysis Parameters Display
        analysis_params_layout = QHBoxLayout()
        self.label_thresh_display = QLabel(f"Analysis Silence Threshold: -30.0 dBFS")
        self.label_min_silence_display = QLabel(f"Min Silence Duration Filter: 150 ms") # Show filter used
        analysis_params_layout.addWidget(self.label_thresh_display)
        analysis_params_layout.addSpacing(20)
        analysis_params_layout.addWidget(self.label_min_silence_display)
        analysis_params_layout.addStretch() # Push to left
        main_layout.addLayout(analysis_params_layout)


        # Analyze Button
        analyze_btn = QPushButton("Analyze and Compare")
        analyze_btn.clicked.connect(self.run_analysis)
        main_layout.addWidget(analyze_btn)

        # Results Display
        self.results_textedit = QTextEdit()
        self.results_textedit.setReadOnly(True)
        self.results_textedit.setFontFamily("monospace") # Use monospace for alignment
        self.results_textedit.setLineWrapMode(QTextEdit.NoWrap) # Prevent wrapping
        self.results_textedit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.results_textedit)

        self.setLayout(main_layout)

    def browse_file1(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select First WAV File', '', 'WAV Files (*.wav)')
        if fname:
            self.file1_path = fname
            self.path1_edit.setText(fname)

    def browse_file2(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Second WAV File', '', 'WAV Files (*.wav)')
        if fname:
            self.file2_path = fname
            self.path2_edit.setText(fname)

    def run_analysis(self):
        if not self.file1_path or not self.file2_path:
            QMessageBox.warning(self, "Missing Files", "Please select both audio files.")
            return
        if not os.path.exists(self.file1_path):
            QMessageBox.warning(self, "File Not Found", f"File not found: {self.file1_path}")
            return
        if not os.path.exists(self.file2_path):
            QMessageBox.warning(self, "File Not Found", f"File not found: {self.file2_path}")
            return

        # --- Parameters used for this script's analysis ---
        analysis_threshold = -30.0
        analysis_min_silence = 150 # Filter out silences shorter than this (ms)

        self.results_textedit.setText(f"Analyzing...\nFile 1: {os.path.basename(self.file1_path)}\nFile 2: {os.path.basename(self.file2_path)}\nAnalysis Thresh: {analysis_threshold:.1f} dBFS, Min Silence Filter: {analysis_min_silence} ms")
        QApplication.processEvents() # Update UI

        stats1 = analyze_audio(self.file1_path,
                               silence_threshold_dbfs=analysis_threshold,
                               min_meaningful_silence_ms=analysis_min_silence)
        stats2 = analyze_audio(self.file2_path,
                               silence_threshold_dbfs=analysis_threshold,
                               min_meaningful_silence_ms=analysis_min_silence)

        # --- Format results focusing on VAD parameters ---
        output_lines = []
        output_lines.append("--- Audio Analysis Comparison (VAD Focus) ---")
        output_lines.append(f"File 1: {os.path.basename(self.file1_path)}")
        output_lines.append(f"File 2: {os.path.basename(self.file2_path)}")
        output_lines.append(f"(Analysis using Threshold: {analysis_threshold:.1f} dBFS, Min Silence Filter: {analysis_min_silence} ms)")
        output_lines.append("-" * 80) # Wider separator

        if stats1.get("error") or stats2.get("error"):
            if stats1.get("error"): output_lines.append(f"Error File 1: {stats1['error']}")
            if stats2.get("error"): output_lines.append(f"Error File 2: {stats2['error']}")
        else:
            # Metrics relevant to VAD tuning
            metrics_to_display = [
                # Loudness Metrics
                "peak_dbfs",
                "speech_rms_dbfs (in speech)",
                "noise_rms_dbfs (in silence)",
                "speech_minus_noise_db",
                # Silence Duration Metrics (using filtered segments)
                "num_meaningful_silent_segments",
                "min_silence_duration_ms",
                "avg_silence_duration_ms",
                "median_silence_duration_ms",
                "max_silence_duration_ms",
                # General Info
                "duration_sec",
                "clipping_percent",
            ]

            # Determine column widths
            max_val1_len = 15 # Initial width for file 1 values
            max_val2_len = 15 # Initial width for file 2 values
            for metric in metrics_to_display:
                 val1 = stats1.get(metric, 'N/A'); val2 = stats2.get(metric, 'N/A')
                 # Format floats with 1 decimal place for dB, 0 for ms/count/percent
                 if "db" in metric or "sec" in metric:
                     # *** FIX HERE ***
                     val1_str = f"{val1:.1f}" if isinstance(val1, (float, np.float64)) else str(val1)
                     val2_str = f"{val2:.1f}" if isinstance(val2, (float, np.float64)) else str(val2)
                 elif "ms" in metric or "num_" in metric or "_percent" in metric:
                     # *** FIX HERE ***
                     val1_str = f"{val1:.0f}" if isinstance(val1, (float, np.float64)) else str(val1)
                     val2_str = f"{val2:.0f}" if isinstance(val2, (float, np.float64)) else str(val2)
                 else:
                      val1_str = str(val1)
                      val2_str = str(val2)

                 max_val1_len = max(max_val1_len, len(val1_str))
                 max_val2_len = max(max_val2_len, len(val2_str))

            # Print header
            metric_col_width = 35 # Width for the metric name column
            header = f"{'Metric':<{metric_col_width}} | {'File 1':<{max_val1_len}} | {'File 2':<{max_val2_len}}"
            output_lines.append(header)
            output_lines.append("-" * len(header))

            # Print metric values
            for metric in metrics_to_display:
                val1 = stats1.get(metric, 'N/A'); val2 = stats2.get(metric, 'N/A')
                # Repeat formatting logic for consistency
                if "db" in metric or "sec" in metric:
                     # *** FIX HERE ***
                     val1_str = f"{val1:.1f}" if isinstance(val1, (float, np.float64)) else str(val1)
                     val2_str = f"{val2:.1f}" if isinstance(val2, (float, np.float64)) else str(val2)
                elif "ms" in metric or "num_" in metric or "_percent" in metric:
                     # *** FIX HERE ***
                     val1_str = f"{val1:.0f}" if isinstance(val1, (float, np.float64)) else str(val1)
                     val2_str = f"{val2:.0f}" if isinstance(val2, (float, np.float64)) else str(val2)
                else:
                      val1_str = str(val1)
                      val2_str = str(val2)
                output_lines.append(f"{metric:<{metric_col_width}} | {val1_str:<{max_val1_len}} | {val2_str:<{max_val2_len}}")

        self.results_textedit.setText("\n".join(output_lines))


# --- Main execution block ---
if __name__ == '__main__':
    try:
        import soundfile
        import numpy
        import statistics
        from PyQt5.QtWidgets import QApplication
    except ImportError as e:
        print(f"Error: Missing required library ({e}). Please install using pip:")
        print(f"  pip install PyQt5 numpy soundfile statistics") # Added statistics
        sys.exit(1)

    app = QApplication(sys.argv)
    analyzer_gui = AudioAnalyzerGUI()
    analyzer_gui.show()
    sys.exit(app.exec_())
# --- END OF FILE analyze_audio.py ---