"""
Training GUI for Paralysis Detection Model Training

Provides an intuitive Tkinter interface for training facial paralysis detection models
with hardware auto-detection, preset scenarios, and smart file detection.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, font as tkfont
import os
import sys
import multiprocessing
import threading
import logging
from pathlib import Path
import queue

# Import training modules
try:
    from hardware_detection import detect_hardware, format_hardware_info
    from paralysis_config import ZONE_CONFIG, INPUT_FILES, MODEL_DIR, LOG_DIR, ANALYSIS_DIR
    from paralysis_training_pipeline import run_zone_training_pipeline
    from paralysis_utils import PARALYSIS_MAP

    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    print(f"Warning: Could not import training modules: {e}")
    ZONE_CONFIG = {}
    INPUT_FILES = {}
    PARALYSIS_MAP = {0: 'Normal', 1: 'Partial', 2: 'Complete'}


class GUILogHandler(logging.Handler):
    """Custom logging handler that sends log messages to the GUI"""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        """Send log record to queue for GUI display"""
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)


def run_zone_training_pipeline_with_progress(zone_key, zone_config_all, input_files_all,
                                             class_names_global, progress_callback):
    """
    Wrapper for run_zone_training_pipeline that reports progress.

    Monitors actual training progress by parsing log output in real-time.

    Args:
        zone_key: Zone identifier ('upper', 'mid', 'lower')
        zone_config_all: Full zone configuration dict
        input_files_all: Input files dict
        class_names_global: Class names mapping
        progress_callback: Callback function(phase, phase_progress, message)
    """
    import time
    import threading
    import re
    from pathlib import Path

    zone_config = zone_config_all[zone_key]
    zone_name = zone_config.get('name', zone_key.capitalize())

    # Get Optuna configuration
    optuna_enabled = zone_config.get('training', {}).get('hyperparameter_tuning', {}).get('enabled', False)
    optuna_trials = zone_config.get('training', {}).get('hyperparameter_tuning', {}).get('optuna', {}).get('n_trials', 100)

    # Get log file path for this zone
    log_file_path = zone_config.get('filenames', {}).get('training_log')

    # Shared state for progress tracking
    progress_state = {
        'phase': 'data_prep',
        'phase_progress': 0,
        'current_trial': 0,
        'total_trials': optuna_trials if optuna_enabled else 0,
        'stop': False
    }

    def monitor_log_progress():
        """Monitor log file for real progress indicators"""
        if not log_file_path:
            return

        # Wait up to 10 seconds for log file to be created
        log_path = Path(log_file_path)
        for _ in range(20):
            if log_path.exists():
                break
            time.sleep(0.5)

        if not log_path.exists():
            return

        phase_names = {
            'data_prep': 'Preparing data',
            'feature_selection': 'Selecting features',
            'optuna': f'Optimizing hyperparameters',
            'training': 'Training final model',
            'shap': 'Computing SHAP explanations',
            'summary': 'Generating summaries'
        }

        try:
            with open(log_file_path, 'r') as f:
                # Start from beginning to catch all messages
                f.seek(0, 0)

                while not progress_state['stop']:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue

                    # Parse progress indicators from logs
                    # Look for: "Starting Optuna", "Optuna Trial X/Y", "Training final model", etc.

                    if 'Preparing data' in line or 'prepare_data' in line.lower():
                        progress_state['phase'] = 'data_prep'
                        progress_callback('data_prep', 10, 'Preparing data')

                    elif 'feature selection' in line.lower() or 'Selecting features' in line:
                        progress_state['phase'] = 'feature_selection'
                        progress_callback('feature_selection', 50, 'Selecting features')

                    elif 'Starting Optuna' in line or 'Running Optuna' in line:
                        progress_state['phase'] = 'optuna'
                        progress_callback('optuna', 0,
                                        f'Starting hyperparameter optimization ({optuna_trials} trials)')

                    # Match: "Optuna Trial 10/100" or similar
                    trial_match = re.search(r'Trial (\d+)/(\d+)', line)
                    if trial_match and progress_state['phase'] == 'optuna':
                        current = int(trial_match.group(1))
                        total = int(trial_match.group(2))
                        progress_state['current_trial'] = current
                        progress_state['total_trials'] = total

                        trial_progress = (current / total * 100) if total > 0 else 0
                        progress_callback('optuna', trial_progress,
                                        f'Optimizing hyperparameters (Trial {current}/{total})')

                    elif 'Training final' in line or 'Fitting main' in line:
                        progress_state['phase'] = 'training'
                        progress_callback('training', 50, 'Training final model')

                    elif 'Computing SHAP' in line or 'SHAP analysis' in line:
                        progress_state['phase'] = 'shap'
                        progress_callback('shap', 50, 'Computing SHAP explanations')

                    elif 'Generating summaries' in line or 'Generating error analysis' in line:
                        progress_state['phase'] = 'summary'
                        progress_callback('summary', 50, 'Generating summaries')

                    elif 'Training pipeline finished' in line:
                        progress_callback('summary', 100, 'Training complete')
                        progress_state['stop'] = True

        except Exception as e:
            print(f"Log monitoring error: {e}")

    # Start log monitoring thread
    monitor_thread = threading.Thread(target=monitor_log_progress, daemon=True)
    monitor_thread.start()

    # Run the actual training pipeline
    try:
        progress_callback('data_prep', 0, f"Starting training for {zone_name}")
        summary_result = run_zone_training_pipeline(zone_key, zone_config_all, input_files_all, class_names_global)

        # Training complete
        progress_state['stop'] = True
        progress_callback('summary', 100, "Training complete")

        return summary_result

    except Exception:
        progress_state['stop'] = True
        raise

    finally:
        # Signal monitoring thread to stop
        progress_state['stop'] = True
        monitor_thread.join(timeout=1.0)


def run_training_in_process(zones_to_train, scenario_name, results_file, expert_file, message_queue):
    """
    Standalone function to run training in a separate process.
    This avoids GIL contention and keeps the GUI responsive.

    Args:
        zones_to_train: List of zone keys to train
        scenario_name: Name of the training scenario
        results_file: Path to results CSV
        expert_file: Path to expert key CSV
        message_queue: multiprocessing.Queue for sending messages back to GUI
    """
    import os
    import sys
    import logging
    import io

    # Redirect stdout/stderr to capture all output
    class QueueWriter:
        def __init__(self, queue, msg_type='log'):
            self.queue = queue
            self.msg_type = msg_type
            self.buffer = ''

        def write(self, text):
            if text:
                self.buffer += text
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line.strip():
                        try:
                            self.queue.put((self.msg_type, line))
                        except:
                            pass

        def flush(self):
            if self.buffer.strip():
                try:
                    self.queue.put((self.msg_type, self.buffer))
                except:
                    pass
                self.buffer = ''

    # Redirect stdout and stderr
    sys.stdout = QueueWriter(message_queue, 'log')
    sys.stderr = QueueWriter(message_queue, 'log')

    # Re-import modules in the new process
    try:
        from paralysis_config import ZONE_CONFIG, INPUT_FILES
        from paralysis_training_pipeline import run_zone_training_pipeline
        from paralysis_utils import PARALYSIS_MAP
    except ImportError as e:
        message_queue.put(('error', f'Failed to import modules: {e}'))
        return

    # Set up logging to also use our queue
    class QueueHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                message_queue.put(('log', msg))
            except Exception:
                pass

    # Add queue handler to root logger (in addition to stdout redirect)
    queue_handler = QueueHandler()
    queue_handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(queue_handler)
    logging.getLogger().setLevel(logging.INFO)

    # Get scenario config
    scenarios = {
        'Development': {'optuna_trials': {'upper': 50, 'mid': 50, 'lower': 80}},
        'Production': {'optuna_trials': {'upper': 200, 'mid': 200, 'lower': 200}},
        'Incomplete-Focus': {'optuna_trials': {'upper': 200, 'mid': 200, 'lower': 200}},
        'Quick-Retrain': {'optuna_trials': None},
    }
    scenario = scenarios.get(scenario_name, scenarios['Production'])

    # Update INPUT_FILES
    INPUT_FILES['results_csv'] = os.path.expanduser(results_file)
    INPUT_FILES['expert_key_csv'] = os.path.expanduser(expert_file)

    message_queue.put(('status', f'Starting training with scenario: {scenario_name}'))
    message_queue.put(('status', f'Zones: {", ".join(zones_to_train)}'))

    # Train each zone
    for zone_idx, zone_key in enumerate(zones_to_train):
        zone_config = ZONE_CONFIG.get(zone_key, {})
        zone_name = zone_config.get('name', zone_key.capitalize())

        message_queue.put(('zone_start', {'zone_idx': zone_idx, 'zone_key': zone_key, 'zone_name': zone_name, 'total': len(zones_to_train)}))

        try:
            # Apply scenario configuration
            if scenario['optuna_trials'] is not None and zone_key in scenario['optuna_trials']:
                if 'training' not in zone_config:
                    zone_config['training'] = {}
                if 'hyperparameter_tuning' not in zone_config['training']:
                    zone_config['training']['hyperparameter_tuning'] = {}
                if 'optuna' not in zone_config['training']['hyperparameter_tuning']:
                    zone_config['training']['hyperparameter_tuning']['optuna'] = {}
                zone_config['training']['hyperparameter_tuning']['enabled'] = True
                zone_config['training']['hyperparameter_tuning']['optuna']['n_trials'] = scenario['optuna_trials'][zone_key]
            elif scenario['optuna_trials'] is None:
                if 'training' in zone_config and 'hyperparameter_tuning' in zone_config['training']:
                    zone_config['training']['hyperparameter_tuning']['enabled'] = False

            # Run training
            summary_result = run_zone_training_pipeline(zone_key, ZONE_CONFIG, INPUT_FILES, PARALYSIS_MAP)

            if summary_result:
                message_queue.put(('zone_complete', {'zone_idx': zone_idx, 'zone_key': zone_key, 'zone_name': zone_name, 'summary': summary_result}))
            else:
                message_queue.put(('zone_complete', {'zone_idx': zone_idx, 'zone_key': zone_key, 'zone_name': zone_name, 'summary': None}))

        except Exception as e:
            import traceback
            message_queue.put(('zone_error', {'zone_idx': zone_idx, 'zone_key': zone_key, 'zone_name': zone_name, 'error': str(e), 'traceback': traceback.format_exc()}))

    message_queue.put(('complete', 'All training completed'))


# Training scenario presets
TRAINING_SCENARIOS = {
    'Development': {
        'description': 'Quick training for development and testing (fewer trials, faster)',
        'optuna_trials': {'upper': 50, 'mid': 50, 'lower': 80},
        'feature_selection': True,
        'shap_analysis': False,  # Skip SHAP for speed during development
        'smote_enabled': True,
        'calibration_enabled': True,
        'threshold_optimization': True
    },
    'Production': {
        'description': 'Full training for production deployment (maximum quality)',
        'optuna_trials': {'upper': 200, 'mid': 200, 'lower': 200},
        'feature_selection': True,
        'shap_analysis': True,
        'smote_enabled': True,
        'calibration_enabled': True,
        'threshold_optimization': True
    },
    'Incomplete-Focus': {
        'description': 'Optimized for detecting incomplete paralysis (class 1)',
        'optuna_trials': {'upper': 200, 'mid': 200, 'lower': 200},
        'feature_selection': True,
        'shap_analysis': True,
        'smote_enabled': True,
        'smote_focus_class': 1,  # Focus SMOTE on partial paralysis
        'calibration_enabled': True,
        'threshold_optimization': True
    },
    'Quick-Retrain': {
        'description': 'Fast retraining with existing hyperparameters (no Optuna)',
        'optuna_trials': None,  # Disable Optuna
        'feature_selection': True,
        'shap_analysis': True,
        'smote_enabled': True,
        'calibration_enabled': True,
        'threshold_optimization': True
    }
}


class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Paralysis Model Training")
        self.root.geometry("950x850")

        # Handle window close to kill any running training process
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Hardware info will be detected asynchronously
        self.hardware_info = {}

        # Setup fonts (monospaced for metrics, standard for UI)
        try:
            # Try macOS/modern fonts first
            self.mono_font = tkfont.Font(family="SF Mono", size=11)
            self.mono_font_small = tkfont.Font(family="SF Mono", size=9)
        except:
            # Fallback to cross-platform fonts
            try:
                self.mono_font = tkfont.Font(family="Consolas", size=11)
                self.mono_font_small = tkfont.Font(family="Consolas", size=9)
            except:
                self.mono_font = tkfont.Font(family="Courier", size=11)
                self.mono_font_small = tkfont.Font(family="Courier", size=9)

        self.header_font = tkfont.Font(family="Helvetica", size=18, weight="normal")
        self.section_font = tkfont.Font(family="Helvetica", size=11, weight="bold")
        self.body_font = tkfont.Font(family="Helvetica", size=11)

        # Variables
        self.selected_zones = {}
        self.scenario_var = tk.StringVar(value='Production')
        self.results_file_var = tk.StringVar()
        self.expert_file_var = tk.StringVar()
        self.training_thread = None
        self.training_process = None
        self.mp_queue = None

        # Progress tracking
        self.total_zones = 0
        self.current_zone_index = 0
        self.current_phase_progress = 0.0
        self.current_phase = ''
        self.current_zone_name = ''
        self.zones_to_train = []  # Track actual zones being trained

        # Log queue and handler for capturing training output
        self.log_queue = queue.Queue()
        self.gui_log_handler = None
        self.log_poll_id = None

        # Summary tracking for Summary tab
        self.zone_summaries = []

        # Setup custom styles (do this early)
        self._setup_styles()

        # Build GUI first for responsiveness
        self.create_widgets()

        # Process pending events to make GUI responsive immediately
        self.root.update_idletasks()

        # Defer slow operations to keep GUI responsive at startup
        self.root.after(50, self._deferred_init)

    def _deferred_init(self):
        """Perform slow initialization after GUI is visible and responsive"""
        # Auto-detect files
        self.auto_detect_files()

        # Detect hardware in background
        if MODULES_LOADED:
            self.hardware_info = detect_hardware()
            # Update hardware tab if it exists
            self.root.update_idletasks()

        # Enable start button and update status now that initialization is complete
        self.start_button.config(state='normal')
        self.progress_var.set("Ready to train")

    def _setup_styles(self):
        """Setup custom ttk styles"""
        style = ttk.Style()
        try:
            style.theme_use('clam')  # Modern look
        except:
            pass  # Use default if clam not available

        # Thicker progress bar (24px instead of default ~20px)
        style.configure("Thick.Horizontal.TProgressbar",
                       thickness=24)

    def auto_detect_files(self):
        """Auto-detect data files if they exist in expected locations"""
        home_dir = os.path.expanduser('~')

        # Priority order for results file detection
        common_results_paths = [
            os.path.join(home_dir, 'Documents/SplitFace/S3O Results/combined_results.csv'),  # Primary default
            os.path.join(os.path.dirname(__file__), '../S3O Results/combined_results.csv'),
            os.path.join(os.path.dirname(__file__), 'combined_results.csv'),
        ]

        # Check common locations first
        for path in common_results_paths:
            if os.path.exists(path):
                # Display as ~/Documents/... for readability
                display_path = path.replace(home_dir, '~')
                self.results_file_var.set(display_path)
                break

        # If not found, try INPUT_FILES config as fallback
        if not self.results_file_var.get() and MODULES_LOADED and INPUT_FILES:
            if 'results_csv' in INPUT_FILES:
                results_path = INPUT_FILES['results_csv']
                if os.path.exists(results_path):
                    display_path = results_path.replace(home_dir, '~')
                    self.results_file_var.set(display_path)

        # Auto-detect expert key file (FPRS FP Key)
        common_expert_paths = [
            os.path.join(home_dir, 'Documents/SplitFace/FPRS FP Key.csv'),
            os.path.join(home_dir, 'Documents/SplitFace/S3O Results/FPRS FP Key.csv'),
            os.path.join(os.path.dirname(__file__), '../FPRS FP Key.csv'),
            os.path.join(os.path.dirname(__file__), 'FPRS FP Key.csv'),
        ]

        for path in common_expert_paths:
            if os.path.exists(path):
                display_path = path.replace(home_dir, '~')
                self.expert_file_var.set(display_path)
                break

        # If not found, try INPUT_FILES config as fallback
        if not self.expert_file_var.get() and MODULES_LOADED and INPUT_FILES:
            if 'expert_key_csv' in INPUT_FILES:
                expert_path = INPUT_FILES['expert_key_csv']
                if os.path.exists(expert_path):
                    display_path = expert_path.replace(home_dir, '~')
                    self.expert_file_var.set(display_path)

    def create_widgets(self):
        """Create all GUI widgets"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Training Configuration
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text='Training Configuration')
        self.create_config_tab(config_frame)

        # Tab 2: Hardware Info
        hardware_frame = ttk.Frame(notebook)
        notebook.add(hardware_frame, text='Hardware Info')
        self.create_hardware_tab(hardware_frame)

        # Tab 3: Training Log
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text='Training Log')
        self.create_log_tab(log_frame)

        # Tab 4: Summary
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text='Summary')
        self.create_summary_tab(summary_frame)

    def create_config_tab(self, parent):
        """Create training configuration tab"""
        # File selection section
        file_frame = ttk.LabelFrame(parent, text="Data Files", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(file_frame, text="Results CSV:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(file_frame, textvariable=self.results_file_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_results_file).grid(row=0, column=2)

        ttk.Label(file_frame, text="Expert CSV:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(file_frame, textvariable=self.expert_file_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_expert_file).grid(row=1, column=2)

        # Training scenario section
        scenario_frame = ttk.LabelFrame(parent, text="Training Scenario", padding=10)
        scenario_frame.pack(fill='x', padx=10, pady=5)

        for scenario_name, scenario_config in TRAINING_SCENARIOS.items():
            rb = ttk.Radiobutton(
                scenario_frame,
                text=f"{scenario_name}: {scenario_config['description']}",
                variable=self.scenario_var,
                value=scenario_name
            )
            rb.pack(anchor='w', pady=2)

        # Zone selection section
        zone_frame = ttk.LabelFrame(parent, text="Zones to Train", padding=10)
        zone_frame.pack(fill='x', padx=10, pady=5)

        if MODULES_LOADED and ZONE_CONFIG:
            for zone_key, zone_config in ZONE_CONFIG.items():
                var = tk.BooleanVar(value=True)
                self.selected_zones[zone_key] = var
                zone_name = zone_config.get('name', zone_key.capitalize())
                ttk.Checkbutton(zone_frame, text=zone_name, variable=var).pack(anchor='w')
        else:
            ttk.Label(zone_frame, text="[WARNING] Could not load zone configuration", foreground='red').pack()

        # Training control section
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', padx=10, pady=10)

        self.start_button = ttk.Button(
            control_frame,
            text="Start Training",
            command=self.start_training,
            state='disabled'  # Disabled until initialization completes
        )
        self.start_button.pack(side='left', padx=5)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Training",
            command=self.stop_training,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)

        # Progress section with improved layout
        progress_frame = ttk.LabelFrame(parent, text="Training Progress", padding=10)
        progress_frame.pack(fill='x', padx=10, pady=(10, 5))

        # Overall progress label with monospaced font
        self.progress_var = tk.StringVar(value="Initializing...")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_var,
                                   font=self.mono_font)
        progress_label.pack(fill='x', pady=(0, 5))

        # Thick progress bar (24px)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate',
                                           maximum=100, style="Thick.Horizontal.TProgressbar")
        self.progress_bar.pack(fill='x', pady=(0, 5))

        # Progress percentage
        self.progress_pct_var = tk.StringVar(value="0%")
        progress_pct = ttk.Label(progress_frame, textvariable=self.progress_pct_var,
                                font=self.mono_font_small)
        progress_pct.pack(anchor='w')

        # Info box below progress bar (detailed phase information)
        info_frame = ttk.LabelFrame(parent, text="Current Phase Details", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)

        # Create info labels with monospaced font for data
        self.info_zone_var = tk.StringVar(value="Zone: --")
        self.info_phase_var = tk.StringVar(value="Phase: --")
        self.info_detail_var = tk.StringVar(value="Status: Ready")
        self.info_hardware_var = tk.StringVar(value="Hardware: --")

        for var in [self.info_zone_var, self.info_phase_var,
                   self.info_detail_var, self.info_hardware_var]:
            label = ttk.Label(info_frame, textvariable=var, font=self.mono_font_small)
            label.pack(fill='x', pady=2)

    def create_hardware_tab(self, parent):
        """Create hardware information tab"""
        # Hardware info display
        info_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, width=80, height=25, font=('Courier', 9))
        info_text.pack(fill='both', expand=True, padx=10, pady=10)

        if self.hardware_info:
            info_text.insert('1.0', format_hardware_info(self.hardware_info))
        else:
            info_text.insert('1.0', "Hardware detection not available")

        info_text.config(state='disabled')

        # Refresh button
        ttk.Button(parent, text="Refresh Hardware Info", command=lambda: self.refresh_hardware(info_text)).pack(pady=5)

    def create_log_tab(self, parent):
        """Create training log tab"""
        self.log_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, width=80, height=30, font=('Courier', 9))
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Clear log button
        ttk.Button(parent, text="Clear Log", command=lambda: self.log_text.delete('1.0', tk.END)).pack(pady=5)

    def create_summary_tab(self, parent):
        """Create summary tab for training and performance summaries"""
        self.summary_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, width=80, height=30, font=('Courier', 9))
        self.summary_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Clear summary button
        ttk.Button(parent, text="Clear Summary", command=self.clear_summary).pack(pady=5)

    def clear_summary(self):
        """Clear the summary display and internal list"""
        self.summary_text.delete('1.0', tk.END)
        self.zone_summaries = []

    def add_zone_summary(self, zone_name, training_summary, performance_summary):
        """Add a zone's summary to the Summary tab"""
        self.zone_summaries.append({
            'zone_name': zone_name,
            'training_summary': training_summary,
            'performance_summary': performance_summary
        })

        # Add to summary text widget
        separator = "=" * 80
        self.summary_text.insert(tk.END, f"\n{separator}\n")
        self.summary_text.insert(tk.END, f"{zone_name.upper()} - TRAINING SUMMARY\n")
        self.summary_text.insert(tk.END, f"{separator}\n")
        self.summary_text.insert(tk.END, training_summary + "\n\n")

        self.summary_text.insert(tk.END, f"{separator}\n")
        self.summary_text.insert(tk.END, f"{zone_name.upper()} - PERFORMANCE SUMMARY\n")
        self.summary_text.insert(tk.END, f"{separator}\n")
        self.summary_text.insert(tk.END, performance_summary + "\n\n")

        # Scroll to end
        self.summary_text.see(tk.END)
        self.root.update_idletasks()

    def refresh_hardware(self, text_widget):
        """Refresh hardware information"""
        self.hardware_info = detect_hardware()
        text_widget.config(state='normal')
        text_widget.delete('1.0', tk.END)
        text_widget.insert('1.0', format_hardware_info(self.hardware_info))
        text_widget.config(state='disabled')

    def browse_results_file(self):
        """Browse for results CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Results CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.results_file_var.get()) if self.results_file_var.get() else os.path.expanduser('~')
        )
        if filename:
            self.results_file_var.set(filename)

    def browse_expert_file(self):
        """Browse for expert CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Expert Key CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.expert_file_var.get()) if self.expert_file_var.get() else os.path.expanduser('~')
        )
        if filename:
            self.expert_file_var.set(filename)

    def log_message(self, message):
        """Add message to training log"""
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def poll_log_queue(self):
        """Poll the log queue and update GUI log display (called periodically)"""
        # Process all available log messages
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + '\n')
                self.log_text.see(tk.END)
            except queue.Empty:
                break

        # Schedule next poll if training is still running
        if self.log_poll_id is not None:
            self.log_poll_id = self.root.after(100, self.poll_log_queue)  # Poll every 100ms

    def poll_subprocess_output(self):
        """Poll subprocess stdout and display in GUI (non-blocking)"""
        try:
            import select
            import re

            if self.training_process and self.training_process.stdout:
                # Use select to check if data is available (non-blocking)
                # Timeout of 0 means don't wait - return immediately
                readable, _, _ = select.select([self.training_process.stdout], [], [], 0)

                lines_processed = 0
                max_lines_per_poll = 20  # Limit lines per poll to keep GUI responsive

                while readable and lines_processed < max_lines_per_poll:
                    line = self.training_process.stdout.readline()
                    if not line:
                        break  # EOF or no more data

                    line = line.rstrip('\n\r')
                    if line:
                        self.log_text.insert(tk.END, line + '\n')
                        self.log_text.see(tk.END)
                        lines_processed += 1

                        # Update progress based on log content
                        # Detect zone changes (e.g., "Starting Training Pipeline for Zone: Upper Face")
                        zone_match = re.search(r'Starting Training Pipeline for Zone:\s*(\w+)', line, re.IGNORECASE)
                        if zone_match:
                            zone_name = zone_match.group(1).capitalize()
                            self.current_zone_name = zone_name
                            # Calculate zone index
                            zone_map = {'upper': 0, 'mid': 1, 'lower': 2}
                            zone_key = zone_name.lower()
                            if zone_key in zone_map and hasattr(self, 'zones_to_train'):
                                # Find position in zones_to_train list
                                for idx, z in enumerate(self.zones_to_train):
                                    if z.lower() == zone_key:
                                        self.current_zone_index = idx
                                        break
                            zone_progress = f"Zone {self.current_zone_index + 1}/{self.total_zones}: {zone_name} Face"
                            self.info_zone_var.set(zone_progress)
                            self.progress_var.set(f"Training {zone_name} Face...")

                        if 'Optuna Trial' in line or 'Trial ' in line:
                            match = re.search(r'Trial (\d+)/(\d+)', line)
                            if match:
                                current, total = int(match.group(1)), int(match.group(2))
                                # Calculate overall progress including zone progress
                                zone_progress_pct = (current / total) * 100
                                if self.total_zones > 0:
                                    overall = ((self.current_zone_index * 100) + zone_progress_pct) / self.total_zones
                                else:
                                    overall = zone_progress_pct
                                self.progress_bar['value'] = overall
                                self.progress_pct_var.set(f"{overall:.1f}%")
                                self.info_detail_var.set(f"Optuna Trial {current}/{total}")
                                zone_info = f" ({self.current_zone_name})" if hasattr(self, 'current_zone_name') and self.current_zone_name else ""
                                self.progress_var.set(f"Optimizing{zone_info}: Trial {current}/{total}")
                        elif 'Training pipeline finished' in line or 'pipeline finished' in line.lower():
                            # Zone completed
                            if self.total_zones > 0:
                                completed = self.current_zone_index + 1
                                overall = (completed / self.total_zones) * 100
                                self.progress_bar['value'] = overall
                                self.progress_pct_var.set(f"{overall:.1f}%")
                            self.info_phase_var.set("Phase: Complete")
                        elif 'Starting Training Pipeline' in line and 'for' not in line:
                            self.info_phase_var.set("Phase: Training")
                            self.progress_var.set("Training in progress...")
                        elif 'feature selection' in line.lower():
                            self.info_phase_var.set("Phase: Feature Selection")
                            self.progress_var.set("Selecting features...")
                        elif 'SMOTE' in line or 'resampling' in line.lower():
                            self.info_phase_var.set("Phase: Data Balancing")
                            self.progress_var.set("Balancing training data...")
                        elif 'Fitting final' in line or 'Training final' in line:
                            self.info_phase_var.set("Phase: Final Model")
                            self.progress_var.set("Training final model...")

                    # Check if more data available
                    readable, _, _ = select.select([self.training_process.stdout], [], [], 0)

        except Exception as e:
            pass  # Ignore read errors (e.g., process terminated)

        # Check if process is still running
        if self.training_process and self.training_process.poll() is None:
            # Process still running, schedule next poll
            self.log_poll_id = self.root.after(50, self.poll_subprocess_output)
        else:
            # Process finished - drain any remaining output
            try:
                if self.training_process and self.training_process.stdout:
                    remaining = self.training_process.stdout.read()
                    if remaining:
                        for line in remaining.splitlines():
                            if line.strip():
                                self.log_text.insert(tk.END, line + '\n')
                        self.log_text.see(tk.END)
            except:
                pass

            exit_code = self.training_process.returncode if self.training_process else -1
            self.log_message(f"\n{'='*60}")
            self.log_message(f"Training process finished (exit code: {exit_code})")
            self.log_message(f"{'='*60}")

            if exit_code == 0:
                self.progress_bar['value'] = 100
                self.progress_pct_var.set("100%")
                self.info_phase_var.set("Phase: Complete")
                self.progress_var.set("Training completed successfully!")
                messagebox.showinfo("Training Complete", "Training finished successfully!")
            else:
                messagebox.showwarning("Training", f"Training finished with exit code {exit_code}")

            self.training_finished()

    def poll_process_queue(self):
        """Poll the multiprocessing queue for messages from training process"""
        try:
            # Process all available messages (non-blocking)
            messages_processed = 0
            while messages_processed < 100:  # Limit per poll to keep GUI responsive
                try:
                    msg_type, msg_data = self.mp_queue.get_nowait()
                    messages_processed += 1
                except Exception:
                    break  # Queue empty or error

                if msg_type == 'log':
                    self.log_text.insert(tk.END, msg_data + '\n')
                    self.log_text.see(tk.END)

                elif msg_type == 'status':
                    self.log_message(msg_data)

                elif msg_type == 'zone_start':
                    zone_idx = msg_data['zone_idx']
                    zone_name = msg_data['zone_name']
                    total = msg_data['total']
                    self.current_zone_index = zone_idx
                    self.log_message(f"\n{'='*60}")
                    self.log_message(f"Training Zone {zone_idx + 1}/{total}: {zone_name}")
                    self.log_message(f"{'='*60}")
                    self.info_zone_var.set(f"Zone: {zone_name} ({zone_idx + 1}/{total})")
                    self.info_phase_var.set("Phase: Training")

                elif msg_type == 'zone_complete':
                    zone_name = msg_data['zone_name']
                    summary = msg_data.get('summary')
                    self.log_message(f"[OK] Zone {zone_name} training completed successfully")
                    if summary:
                        self.add_zone_summary(
                            summary['zone_name'],
                            summary['training_summary'],
                            summary['performance_summary']
                        )
                    # Update progress
                    zone_idx = msg_data['zone_idx']
                    progress = ((zone_idx + 1) / self.total_zones) * 100
                    self.progress_bar['value'] = progress
                    self.progress_pct_var.set(f"{progress:.1f}%")

                elif msg_type == 'zone_error':
                    zone_name = msg_data['zone_name']
                    error = msg_data['error']
                    self.log_message(f"[FAILED] Zone {zone_name} training failed: {error}")
                    if 'traceback' in msg_data:
                        self.log_message(msg_data['traceback'])

                elif msg_type == 'error':
                    self.log_message(f"[ERROR] {msg_data}")
                    messagebox.showerror("Training Error", str(msg_data))

                elif msg_type == 'complete':
                    self.log_message(f"\n{'='*60}")
                    self.log_message("ALL TRAINING COMPLETED")
                    self.log_message(f"{'='*60}")
                    self.progress_bar['value'] = 100
                    self.progress_pct_var.set("100%")
                    self.info_phase_var.set("Phase: Complete")
                    self.info_detail_var.set("Status: Training finished successfully")
                    messagebox.showinfo("Training Complete", "All zones have been trained successfully!")
                    self.training_finished()
                    return  # Stop polling

        except Exception as e:
            self.log_message(f"Queue polling error: {e}")

        # Check if process is still running
        if hasattr(self, 'training_process') and self.training_process.is_alive():
            self.log_poll_id = self.root.after(50, self.poll_process_queue)  # Poll every 50ms for responsiveness
        else:
            # Process ended - check if it was successful
            if hasattr(self, 'training_process') and self.training_process.exitcode != 0:
                self.log_message(f"Training process exited with code: {self.training_process.exitcode}")
            self.training_finished()

    def update_progress(self, zone_index, phase, phase_progress, message):
        """
        Update progress bar and status message (thread-safe).

        Args:
            zone_index: Current zone index (0-based)
            phase: Phase name ('data_prep', 'feature_selection', 'optuna', 'training', 'shap', 'summary')
            phase_progress: Progress within current phase (0-100)
            message: Status message to display
        """
        # Phase weights (must sum to 100)
        phase_weights = {
            'data_prep': 10,
            'feature_selection': 15,
            'optuna': 40,
            'training': 20,
            'shap': 10,
            'summary': 5
        }

        # Calculate phase start percentage
        phase_order = ['data_prep', 'feature_selection', 'optuna', 'training', 'shap', 'summary']
        phase_start = sum(phase_weights[p] for p in phase_order[:phase_order.index(phase)])

        # Calculate current zone progress (0-100)
        zone_progress = phase_start + (phase_weights[phase] * phase_progress / 100.0)

        # Calculate overall progress across all zones
        if self.total_zones > 0:
            overall_progress = (zone_index * 100 + zone_progress) / self.total_zones
        else:
            overall_progress = 0

        # Store phase info for info box updates
        self.current_phase = phase
        self.current_zone_index = zone_index

        # Update GUI (thread-safe via root.after)
        def update_gui():
            self.progress_bar['value'] = overall_progress
            self.progress_var.set(message)
            self.progress_pct_var.set(f"{overall_progress:.1f}%")

            # Update info box with detailed phase information
            # Get actual zone name from the zones being trained
            if zone_index < len(self.zones_to_train):
                zone_key = self.zones_to_train[zone_index]
                zone_name = ZONE_CONFIG.get(zone_key, {}).get('name', zone_key.capitalize())
            else:
                zone_name = f"Zone {zone_index + 1}"

            phase_names = {
                'data_prep': 'Data Preparation',
                'feature_selection': 'Feature Selection',
                'optuna': 'Hyperparameter Optimization',
                'training': 'Model Training',
                'shap': 'SHAP Analysis',
                'summary': 'Summary Generation'
            }

            self.info_zone_var.set(f"Zone: {zone_name} ({zone_index + 1}/{self.total_zones})")
            self.info_phase_var.set(f"Phase: {phase_names.get(phase, phase.replace('_', ' ').title())}")
            self.info_detail_var.set(f"Progress: {phase_progress:.1f}% of current phase")

            # Update hardware info
            if self.hardware_info:
                cpu_cores = self.hardware_info.get('cpu_cores_physical', 'N/A')
                processor = self.hardware_info.get('processor', 'Unknown')
                # Shorten processor name
                if 'Apple' in processor and 'M' in processor:
                    # Extract "M3 Max" from "Apple M3 Max"
                    parts = processor.split()
                    if len(parts) >= 2:
                        processor = ' '.join(parts[1:])
                self.info_hardware_var.set(f"Hardware: {processor} ({cpu_cores} cores)")
            else:
                self.info_hardware_var.set("Hardware: Not detected")

        self.root.after(0, update_gui)

    def start_training(self):
        """Start training process"""
        # Expand paths (convert ~ to full path)
        results_path = os.path.expanduser(self.results_file_var.get())
        expert_path = os.path.expanduser(self.expert_file_var.get())

        # Validate inputs
        if not self.results_file_var.get() or not os.path.exists(results_path):
            messagebox.showerror("Error", "Please select a valid results CSV file")
            return

        if not self.expert_file_var.get() or not os.path.exists(expert_path):
            messagebox.showerror("Error", "Please select a valid expert key CSV file")
            return

        zones_to_train = [zone for zone, var in self.selected_zones.items() if var.get()]
        if not zones_to_train:
            messagebox.showerror("Error", "Please select at least one zone to train")
            return

        if not MODULES_LOADED:
            messagebox.showerror("Error", "Training modules could not be loaded. Check imports.")
            return

        # Disable start button, enable stop button
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Initialize progress tracking
        self.total_zones = len(zones_to_train)
        self.zones_to_train = zones_to_train  # Store for zone name lookup
        self.current_zone_index = 0
        self.progress_bar['value'] = 0
        self.progress_pct_var.set("0.0%")

        # Initialize info box
        self.info_zone_var.set("Zone: Preparing...")
        self.info_phase_var.set("Phase: Initializing")
        self.info_detail_var.set("Status: Starting training pipeline")

        # Display hardware info
        if self.hardware_info:
            cpu_cores = self.hardware_info.get('cpu_cores_physical', 'N/A')
            processor = self.hardware_info.get('processor', 'Unknown')
            # Shorten processor name
            if 'Apple' in processor and 'M' in processor:
                parts = processor.split()
                if len(parts) >= 2:
                    processor = ' '.join(parts[1:])
            self.info_hardware_var.set(f"Hardware: {processor} ({cpu_cores} cores)")
        else:
            self.info_hardware_var.set("Hardware: Not detected")

        # Clear log and summary
        self.log_text.delete('1.0', tk.END)
        self.clear_summary()

        # Use subprocess for reliable process isolation
        import subprocess
        import json

        # Build command to run training
        script_dir = os.path.dirname(os.path.abspath(__file__))
        training_script = os.path.join(script_dir, 'paralysis_training_pipeline.py')

        # Prepare environment
        env = os.environ.copy()
        env['PYTHONPATH'] = script_dir

        # Build arguments
        cmd = [
            sys.executable,
            training_script
        ] + zones_to_train

        self.log_message(f"Starting training subprocess...")
        self.log_message(f"Zones: {', '.join(zones_to_train)}")
        self.log_message(f"Scenario: {self.scenario_var.get()}\n")

        # Apply scenario config BEFORE starting subprocess (modifies global ZONE_CONFIG)
        scenario = TRAINING_SCENARIOS[self.scenario_var.get()]
        for zone_key in zones_to_train:
            self.apply_scenario_config(zone_key, scenario)

        # Update INPUT_FILES
        INPUT_FILES['results_csv'] = os.path.expanduser(self.results_file_var.get())
        INPUT_FILES['expert_key_csv'] = os.path.expanduser(self.expert_file_var.get())

        # Start subprocess with stdout/stderr piped
        self.training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd=script_dir
        )

        self.log_message(f"Training process started (PID: {self.training_process.pid})")
        self.log_message(f"Using subprocess for optimal performance\n")

        # Start polling subprocess output
        self.log_poll_id = self.root.after(50, self.poll_subprocess_output)

    def run_training(self, zones_to_train):
        """Run training process in background thread"""
        try:
            scenario = TRAINING_SCENARIOS[self.scenario_var.get()]
            self.log_message(f"Starting training with scenario: {self.scenario_var.get()}")
            self.log_message(f"Zones: {', '.join(zones_to_train)}")
            self.log_message(f"Hardware: {self.hardware_info.get('processor', 'Unknown')}\n")

            # Update INPUT_FILES with selected files (expand ~ to full path)
            INPUT_FILES['results_csv'] = os.path.expanduser(self.results_file_var.get())
            INPUT_FILES['expert_key_csv'] = os.path.expanduser(self.expert_file_var.get())

            # Train each zone
            for zone_idx, zone_key in enumerate(zones_to_train):
                zone_name = ZONE_CONFIG.get(zone_key, {}).get('name', zone_key.capitalize())

                self.log_message(f"\n{'='*60}")
                self.log_message(f"Training Zone {zone_idx + 1}/{len(zones_to_train)}: {zone_name}")
                self.log_message(f"{'='*60}")

                try:
                    # Apply scenario configuration to zone config
                    self.apply_scenario_config(zone_key, scenario)

                    # Create progress callback for this zone
                    def progress_callback(phase, phase_progress, message):
                        self.update_progress(zone_idx, phase, phase_progress,
                                           f"Zone {zone_idx + 1}/{len(zones_to_train)} ({zone_name}) - {message}")
                        # Also log important milestones
                        if phase_progress == 0:
                            self.log_message(f"  Started: {message}")
                        elif phase_progress == 100:
                            self.log_message(f"  [OK] Completed: {message}")

                    # Run zone training with progress callback
                    summary_result = run_zone_training_pipeline_with_progress(
                        zone_key, ZONE_CONFIG, INPUT_FILES, PARALYSIS_MAP,
                        progress_callback
                    )

                    self.log_message(f"[OK] Zone {zone_name} training completed successfully")

                    # Add summary to Summary tab (thread-safe)
                    if summary_result:
                        def add_summary():
                            self.add_zone_summary(
                                summary_result['zone_name'],
                                summary_result['training_summary'],
                                summary_result['performance_summary']
                            )
                        self.root.after(0, add_summary)

                except Exception as e:
                    self.log_message(f"[FAILED] Zone {zone_name} training failed: {e}")
                    import traceback
                    self.log_message(traceback.format_exc())

            self.log_message(f"\n{'='*60}")
            self.log_message("ALL TRAINING COMPLETED")
            self.log_message(f"{'='*60}")
            # Use last zone index (len - 1) to avoid showing "Zone 4 (4/3)"
            final_zone_index = max(0, len(zones_to_train) - 1)
            self.update_progress(final_zone_index, 'summary', 100, "Training completed!")

            self.root.after(0, lambda: messagebox.showinfo("Training Complete",
                                                           "All zones have been trained successfully!"))

        except Exception as e:
            self.log_message(f"\n[FAILED] Training failed with error: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"Training failed: {e}"))

        finally:
            # Re-enable buttons
            self.root.after(0, self.training_finished)

    def apply_scenario_config(self, zone_key, scenario):
        """Apply scenario configuration to zone config"""
        zone_config = ZONE_CONFIG[zone_key]

        # Apply Optuna trials
        if scenario['optuna_trials'] is not None and zone_key in scenario['optuna_trials']:
            if 'training' not in zone_config:
                zone_config['training'] = {}
            if 'hyperparameter_tuning' not in zone_config['training']:
                zone_config['training']['hyperparameter_tuning'] = {}
            if 'optuna' not in zone_config['training']['hyperparameter_tuning']:
                zone_config['training']['hyperparameter_tuning']['optuna'] = {}

            zone_config['training']['hyperparameter_tuning']['enabled'] = True
            zone_config['training']['hyperparameter_tuning']['optuna']['n_trials'] = scenario['optuna_trials'][zone_key]
        elif scenario['optuna_trials'] is None:
            # Disable Optuna for quick retrain
            if 'training' in zone_config and 'hyperparameter_tuning' in zone_config['training']:
                zone_config['training']['hyperparameter_tuning']['enabled'] = False

    def training_finished(self):
        """Called when training finishes"""
        # Stop polling
        if self.log_poll_id is not None:
            self.root.after_cancel(self.log_poll_id)
            self.log_poll_id = None

        # Process any remaining messages from multiprocessing queue
        if self.mp_queue is not None:
            try:
                while True:
                    msg_type, msg_data = self.mp_queue.get_nowait()
                    if msg_type == 'log':
                        self.log_text.insert(tk.END, msg_data + '\n')
                        self.log_text.see(tk.END)
            except:
                pass

        # Process any remaining messages in thread log queue
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + '\n')
                self.log_text.see(tk.END)
            except queue.Empty:
                break

        # Clean up training process (works for both subprocess.Popen and multiprocessing.Process)
        if self.training_process is not None:
            try:
                if hasattr(self.training_process, 'poll'):
                    # subprocess.Popen
                    if self.training_process.poll() is None:
                        self.training_process.terminate()
                        self.training_process.wait(timeout=2)
                elif hasattr(self.training_process, 'is_alive'):
                    # multiprocessing.Process
                    if self.training_process.is_alive():
                        self.training_process.terminate()
                        self.training_process.join(timeout=2)
            except:
                pass
            self.training_process = None

        # Detach log handler
        if self.gui_log_handler is not None:
            logging.getLogger().removeHandler(self.gui_log_handler)
            self.gui_log_handler = None

        # Re-enable controls
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_bar.stop()

    def stop_training(self):
        """Stop training process"""
        if self.training_process is not None and self.training_process.poll() is None:
            if messagebox.askyesno("Stop Training", "Are you sure you want to stop training? This will terminate the process."):
                self.log_message("\n[STOPPING] Training process...")
                self.training_process.terminate()
                try:
                    self.training_process.wait(timeout=5)
                except:
                    self.training_process.kill()
                self.log_message("Training stopped by user.")
                self.training_finished()
        else:
            messagebox.showinfo("Stop Training", "No training is currently running.")

    def on_closing(self):
        """Handle window close - terminate any running training process"""
        if self.training_process is not None:
            # Check if process is still running
            if hasattr(self.training_process, 'poll') and self.training_process.poll() is None:
                # Ask user if they want to stop training
                if messagebox.askyesno("Close", "Training is still running. Stop training and close?"):
                    self.training_process.terminate()
                    try:
                        self.training_process.wait(timeout=3)
                    except:
                        self.training_process.kill()
                else:
                    return  # Don't close

        # Cancel any pending after() callbacks
        if self.log_poll_id is not None:
            self.root.after_cancel(self.log_poll_id)

        self.root.destroy()


def main():
    """Main entry point"""
    # Set multiprocessing start method to 'spawn' for macOS compatibility
    # This ensures clean process isolation and avoids fork-related issues
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # Already set

    root = tk.Tk()

    # Create and run app
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    multiprocessing.freeze_support()
    main()
