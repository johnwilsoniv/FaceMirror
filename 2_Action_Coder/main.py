# main.py - Main application entry point with sequential batch processing
import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer, Qt, QEvent

# Import the enhanced video player instead of the OpenCV player
from qt_media_player import QTMediaPlayer
from gui_component import MainWindow
from action_tracker import ActionTracker
from csv_handler import CSVHandler
from video_processor import VideoProcessor
from player_integration import integrate_qt_player
from batch_processor import BatchProcessor  # Import the new batch processor

def add_exception_logging():
    """Add global exception handler to log unhandled exceptions."""
    import sys
    import traceback
    import datetime
    import os
    
    # Create a logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Original excepthook
    original_excepthook = sys.excepthook
    
    def exception_handler(exc_type, exc_value, exc_traceback):
        """Log unhandled exceptions to a file and show a message box."""
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create a log file with the timestamp
        log_file = os.path.join("logs", f"crash_{timestamp}.log")
        
        # Format the exception
        exception_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Write to log file
        with open(log_file, "w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Exception: {exc_type.__name__}: {exc_value}\n")
            f.write("Traceback:\n")
            f.write(exception_text)
        
        # Call the original excepthook
        original_excepthook(exc_type, exc_value, exc_traceback)
        
        # Show a message box with error info (if PyQt is running)
        try:
            from PyQt5.QtWidgets import QMessageBox, QApplication
            if QApplication.instance():
                QMessageBox.critical(
                    None,
                    "Application Error",
                    f"An unexpected error occurred: {exc_type.__name__}: {exc_value}\n\n"
                    f"The error has been logged to {log_file}"
                )
        except:
            pass
    
    # Set the custom exception hook
    sys.excepthook = exception_handler
    
    print("Exception logging enabled.")

class ProcessingThread(QThread):
    """Thread for processing output files."""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, video_path, action_tracker, csv_handler, output_csv, output_video, output_second_csv=None):
        """Initialize the processing thread."""
        super().__init__()
        self.video_path = video_path
        self.action_tracker = action_tracker
        self.csv_handler = csv_handler
        self.output_csv = output_csv
        self.output_video = output_video
        self.output_second_csv = output_second_csv
    
    def run(self):
        """Run the processing tasks with improved action handling."""
        try:
            # Validate action ranges before processing
            self.progress_signal.emit(5)
            self.action_tracker.validate_ranges()
            
            # Get all actions after validation
            all_actions = self.action_tracker.get_all_actions()
            
            # Add actions to CSV(s)
            self.progress_signal.emit(10)
            if not self.csv_handler.add_action_column(all_actions):
                self.error_signal.emit("Failed to add action column to CSV files.")
                return
                
            self.progress_signal.emit(20)
            if not self.csv_handler.save_data(self.output_csv, self.output_second_csv):
                self.error_signal.emit("Failed to save CSV files.")
                return
                
            self.progress_signal.emit(30)
            
            # Process video
            video_processor = VideoProcessor(self.video_path, self.action_tracker)
            
            # Use a lambda to pass progress from video processor to our signal
            def update_progress(progress):
                # Scale progress from 30-100 range
                scaled_progress = 30 + int(progress * 0.7)
                self.progress_signal.emit(scaled_progress)
            
            if not video_processor.process_video(self.output_video, update_progress):
                self.error_signal.emit("Failed to process video.")
                return
            
            self.progress_signal.emit(100)
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(f"Error during processing: {str(e)}")

class ApplicationController(QObject):
    """Controller class to coordinate between components."""
    
    def __init__(self):
        """Initialize the application controller."""
        super().__init__()
        self.app = QApplication(sys.argv)
        
        # Set application-wide Qt attributes for better responsiveness
        self.app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
        self.app.setAttribute(Qt.AA_Use96Dpi)  # Consistent DPI handling
        
        # Set process priority for better responsiveness on Windows
        try:
            import psutil
            process = psutil.Process()
            process.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows
        except (ImportError, AttributeError, OSError):
            pass  # Silently continue if not on Windows or psutil not available
        
        self.window = MainWindow()
        
        # Use the enhanced QTMediaPlayer instead of OpenCVVideoPlayer
        self.video_player = QTMediaPlayer()
        self.action_tracker = ActionTracker()
        self.csv_handler = CSVHandler()
        self.processing_thread = None
        self.active_action = None  # Track the currently active action
        
        # Initialize the batch processor
        self.batch_processor = BatchProcessor()
        
        # Integrate the QT player into the UI
        integrate_qt_player(self.window, self.video_player)

        # Timer to ensure action state persistence - adaptive refresh rate
        self.state_refresh_timer = QTimer()
        # Default interval of 50ms (20Hz), will be updated based on video FPS
        self.state_refresh_timer.setInterval(50)
        self.state_refresh_timer.timeout.connect(self.refresh_action_state)
        
        # Add a dedicated timer for input handling with higher priority
        self.input_timer = QTimer()
        self.input_timer.setInterval(10)  # 10ms = 100Hz for responsive input
        self.input_timer.timeout.connect(self.process_input_state)
        self.input_timer.start()  # Always keep input processing active
        
        # Connect signals and slots
        self.connect_signals()
        
        self.window.show()
        
        # Apply a special event filter for key events
        self.window.installEventFilter(self)
        
        # Ensure window has keyboard focus
        self.window.activateWindow()
        self.window.setFocus(Qt.OtherFocusReason)

    def eventFilter(self, obj, event):
        """
        Event filter to control key events and prevent multiple activations.
        """
        if obj == self.window:
            if event.type() == QEvent.KeyPress:
                key = event.text()
                
                # If the key is already marked as pressed, block the event
                if key in self.window.key_to_action:
                    is_pressed = self.window.pressed_keys.get(key, False)
                    if is_pressed:
                        return True  # Block the event
                    
        # Let other events be handled normally
        return super().eventFilter(obj, event)

    def process_input_state(self):
        """
        Dedicated method to process input state at high frequency.
        This ensures button/key states are processed even during heavy video playback.
        """
        # If we have an active action, make sure it's being properly applied
        if self.active_action:
            # Get current frame
            current_frame = self.video_player.current_frame
            
            # Only update the action tracker if the frame has changed
            if current_frame != self.action_tracker.current_frame:
                self.action_tracker.set_frame(current_frame)
                # Note: With range-based tracking, we don't need to continuously call continue_action
            
            # Ensure UI state is consistent
            if self.window.current_active_action != self.active_action:
                # Update window state
                self.window.current_active_action = self.active_action
                
                # Ensure button visual state is correct
                if self.active_action in self.window.action_buttons:
                    button = self.window.action_buttons[self.active_action]
                    if not button.pressed_state:
                        button.set_pressed(True)

    def connect_signals(self):
        """Connect all signals and slots."""
        # Main window signals - single file mode
        self.window.play_pause_signal.connect(self.toggle_play_pause)
        self.window.frame_changed_signal.connect(self.seek_to_frame)
        self.window.save_signal.connect(self.save_outputs)
        self.window.action_started_signal.connect(self.start_action)
        self.window.action_stopped_signal.connect(self.stop_action)
        self.window.action_continued_signal.connect(self.continue_action)
        
        # Main window signals - batch mode
        self.window.select_batch_dir_signal.connect(self.scan_batch_directory)
        self.window.next_file_signal.connect(self.load_next_file)
        self.window.previous_file_signal.connect(self.load_previous_file)
        self.window.first_file_signal.connect(self.load_first_file)
        
        # New signal for multi-video selection
        self.window.videos_selected_signal.connect(self.handle_selected_videos)

        # Video player signals
        self.video_player.frameChanged.connect(self.update_frame)
        self.video_player.videoFinished.connect(self.video_finished)

    def handle_selected_videos(self, video_files):
        """
        Handle a list of selected video files.
        Finds matching CSVs and sets up batch navigation.
        
        Args:
            video_files: List of paths to selected video files
        """
        if not video_files:
            return
            
        # Find matching CSV files for the selected videos
        file_sets = self.batch_processor.find_matching_files_for_videos(video_files)
        
        if not file_sets:
            QMessageBox.warning(
                self.window,
                "No Matches Found",
                "No matching CSV files were found for the selected videos."
            )
            return
        
        # Set these file sets in the batch processor
        self.batch_processor.set_file_sets(file_sets)
        
        # Load the first file
        self.load_first_file()

    def continue_action(self):
        """Continue the current action for the current frame."""
        # With range-based tracking, we don't need to continuously call continue_action
        # The action applies to the entire range from start until stop
        pass

    def refresh_action_state(self):
        """
        Refresh action state to ensure continuous activation.
        Now optimized to only perform necessary updates.
        """
        # Skip if we're not in playback or no active action
        if not self.active_action or not self.video_player.is_playing:
            return
            
        # Get current frame only once
        current_frame = self.video_player.current_frame
        
        # Check if we actually need to update the action tracker
        if current_frame != self.action_tracker.current_frame:
            self.action_tracker.set_frame(current_frame)
            # With range-based tracking, we don't need to continuously call continue_action
            
            # Only update UI if needed
            if self.window.current_active_action != self.active_action:
                if self.active_action in self.window.action_buttons:
                    self.window.action_buttons[self.active_action].set_pressed(True)
                self.window.current_active_action = self.active_action
                
            # Only update overlay if needed
            if self.video_player.current_action != self.active_action:
                self.video_player.set_action(self.active_action)

    def init_video_player(self):
        """Initialize the video player with the selected video path."""
        # In batch tab, don't show warnings if no video is selected yet
        if not self.window.video_path:
            # Only show the warning if we're not in the batch tab
            if not self.window.tab_widget.currentIndex() == 1:  # Not the batch tab
                QMessageBox.warning(self.window, "Warning", "Please select a video file first.")
            return False

        # Initialize with new video
        if not self.video_player.set_video_path(self.window.video_path):
            QMessageBox.critical(self.window, "Error", f"Could not open video file: {self.window.video_path}")
            return False

        # Update UI with video information
        video_props = self.video_player.get_video_properties()
        if video_props:
            self.window.set_total_frames(video_props['total_frames'])

            # Update refresh timer based on video FPS (use at most 4 updates per frame)
            fps = video_props['fps']
            if fps > 0:
                # Limit refreshes to 4x per frame or 50ms, whichever is longer
                interval = max(50, int(250 / fps))
                self.state_refresh_timer.setInterval(interval)
                print(f"Action state refresh interval set to {interval}ms based on {fps} FPS")

            return True
        return False
    
    def init_csv_handler(self):
        """Initialize the CSV handler with the selected CSV paths."""
        success = True
        
        if self.window.csv_path:
            if not self.csv_handler.set_input_file(self.window.csv_path):
                QMessageBox.critical(self.window, "Error", f"Could not load CSV file: {self.window.csv_path}")
                success = False
        else:
            QMessageBox.warning(self.window, "Warning", "Please select at least one CSV file first.")
            success = False
            
        # Load the second CSV if it's provided
        if self.window.second_csv_path:
            if not self.csv_handler.set_second_input_file(self.window.second_csv_path):
                QMessageBox.critical(self.window, "Error", f"Could not load second CSV file: {self.window.second_csv_path}")
                success = False
            
        return success

    def update_frame(self, frame_number, qimage, action_code):
        """Update the UI with the current video frame."""
        # Update frame information in the UI
        self.window.update_frame_info(frame_number, self.video_player.total_frames)

        # Update action tracker with current frame
        self.action_tracker.set_frame(frame_number)

        # Determine which action code to display (if any)
        display_action = None

        # Priority 1: Active action (user is currently pressing a button or key)
        if self.active_action:
            display_action = self.active_action

        # Priority 2: Previously recorded action for this frame
        elif not display_action:
            # Get any previously coded action for this frame
            previous_action = self.action_tracker.get_action_for_frame(frame_number)
            if previous_action:
                display_action = previous_action

        # Update the UI action display
        self.window.update_action_display(display_action)

        # Update the video player's overlay - only if needed
        if display_action and display_action != self.video_player.current_action:
            self.video_player.set_action(display_action)
        elif not display_action and self.video_player.current_action:
            self.video_player.clear_action()

    def toggle_play_pause(self, play):
        """Toggle between play and pause states."""
        # If we're in batch mode and no video is selected yet, just update the UI state
        # without showing any warning messages
        if self.window.batch_mode and not self.window.video_path:
            self.window.play_pause_btn.setText("Play")
            return

        # Ensure video is loaded
        if not self.init_video_player():
            self.window.play_pause_btn.setText("Play")
            return

        if play:
            self.video_player.play()
            # Start the state refresh timer only when playing
            self.state_refresh_timer.start()
        else:
            self.video_player.pause()
            # Stop the timer when paused to save resources
            self.state_refresh_timer.stop()

    def seek_to_frame(self, frame):
        """Seek to a specific frame in the video."""
        if not self.init_video_player():
            return

        # Retrieve any previously coded action for this frame
        previous_action = self.action_tracker.get_action_for_frame(frame)

        # Set the action in the video player before seeking to ensure overlay appears
        # Only update if needed to avoid unnecessary redraws
        if previous_action and self.video_player.current_action != previous_action:
            self.video_player.set_action(previous_action)
        elif not previous_action and self.video_player.current_action:
            self.video_player.clear_action()

        # Seek to the frame
        self.video_player.seek(frame)

    def start_action(self, action_code):
        """Start an action at the current frame."""
        # First check if video is loaded
        if not self.window.video_path:
            QMessageBox.warning(self.window, "Warning", "Please select a video file first.")
            # Clear any UI action state
            for code, button in self.window.action_buttons.items():
                button.set_pressed(False)
            self.active_action = None
            self.window.current_active_action = None
            return

        # Check if video player is ready - initialize if needed
        if not hasattr(self.video_player, 'total_frames') or self.video_player.total_frames == 0:
            try:
                if not self.init_video_player():
                    # Clear any UI action state
                    for code, button in self.window.action_buttons.items():
                        button.set_pressed(False)
                    self.active_action = None
                    self.window.current_active_action = None
                    return
            except Exception as e:
                QMessageBox.critical(self.window, "Error", f"Error initializing video: {str(e)}")
                # Clear any UI action state
                for code, button in self.window.action_buttons.items():
                    button.set_pressed(False)
                self.active_action = None
                self.window.current_active_action = None
                return

        try:
            # First clear any current active action
            if self.active_action and self.active_action != action_code:
                self.stop_action()
                
            # Set the active action
            self.active_action = action_code
            self.window.current_active_action = action_code
            
            # Update action tracker with current frame
            current_frame = self.video_player.current_frame
            self.action_tracker.set_frame(current_frame)
            self.action_tracker.start_action(action_code)
            
            # Update overlay display
            if hasattr(self.video_player, 'current_action') and self.video_player.current_action != action_code:
                self.video_player.set_action(action_code)
            
            # Ensure the refresh timer is running while an action is active
            if not self.state_refresh_timer.isActive():
                self.state_refresh_timer.start()
        except Exception as e:
            print(f"Error in start_action: {str(e)}")
            QMessageBox.critical(self.window, "Error", f"Error setting action: {str(e)}")
            # Clear any UI action state
            for code, button in self.window.action_buttons.items():
                button.set_pressed(False)
            self.active_action = None
            self.window.current_active_action = None

    def stop_action(self):
        """Stop the active action."""
        # Save current action and clear active state immediately for responsiveness
        current_action = self.active_action
        self.active_action = None
        self.window.current_active_action = None

        try:
            # Update action tracker with current frame first before stopping action
            current_frame = self.video_player.current_frame
            self.action_tracker.set_frame(current_frame)
            
            # Now stop the action - this will set the end frame for the current action range
            self.action_tracker.stop_action()

            # Clear action in video player
            if hasattr(self.video_player, 'current_action') and self.video_player.current_action:
                self.video_player.clear_action()
            
            # Update UI
            self.window.update_action_display(None)
            
            # If we're not playing, we can stop the refresh timer to save resources
            if not self.video_player.is_playing:
                self.state_refresh_timer.stop()
        except Exception as e:
            # Just log the error, but don't show message since this is called during cleanup
            print(f"Error in stop_action: {str(e)}")

        # Ensure UI state is consistent
        for code, button in self.window.action_buttons.items():
            button.set_pressed(False)
    
    def video_finished(self):
        """Handle video playback completion."""
        self.window.play_pause_btn.setText("Play")
        # Stop the refresh timer when video finishes
        self.state_refresh_timer.stop()

    def save_outputs(self):
        """Generate output files."""
        # Check video loaded
        if not self.init_video_player():
            return

        # Check CSV loaded - avoid using DataFrame in Boolean context
        if self.csv_handler.data is None and not self.init_csv_handler():
            return

        # Create output directory if it doesn't exist
        # If we're in batch mode, create output in the 2.5_Coded_Files directory
        if self.window.batch_mode and self.window.batch_directory:
            # Going one directory up from batch directory
            batch_parent_dir = os.path.dirname(self.window.batch_directory)
            output_dir = os.path.join(batch_parent_dir, "2.5_Coded_Files")
        else:
            # Going one directory up from script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(parent_dir, "2.5_Coded_Files")

        os.makedirs(output_dir, exist_ok=True)

        # Generate output filenames based on input files
        video_basename = os.path.splitext(os.path.basename(self.window.video_path))[0]

        # First CSV output
        csv_basename = os.path.splitext(os.path.basename(self.window.csv_path))[0]
        output_csv = os.path.join(output_dir, f"{csv_basename}_annotated.csv")

        # Second CSV output (if available)
        second_output_csv = None
        if self.window.second_csv_path:
            second_csv_basename = os.path.splitext(os.path.basename(self.window.second_csv_path))[0]
            second_output_csv = os.path.join(output_dir, f"{second_csv_basename}_annotated.csv")

        output_video = os.path.join(output_dir, f"{video_basename}_annotated.mp4")

        # Store output paths in window for progress messages
        self.window.output_csv_path = output_csv
        self.window.output_second_csv_path = second_output_csv
        self.window.output_video_path = output_video

        # Validate that we have some actions to save
        actions = self.action_tracker.get_all_actions()
        if not actions:
            reply = QMessageBox.question(
                self.window,
                "No Actions",
                "No actions have been recorded. Do you want to continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Create processing thread to avoid UI freezing
        self.processing_thread = ProcessingThread(
            self.window.video_path,
            self.action_tracker,
            self.csv_handler,
            self.window.output_csv_path,
            self.window.output_video_path,
            self.window.output_second_csv_path
        )

        # Connect signals
        self.processing_thread.progress_signal.connect(self.window.progress_update_signal.emit)

        # If in batch mode, connect to batch-aware completion handler
        if self.window.batch_mode:
            self.processing_thread.finished_signal.connect(self.batch_file_processed)
            self.processing_thread.error_signal.connect(self.batch_processing_error)
        else:
            # Otherwise use standard completion handler
            self.processing_thread.finished_signal.connect(self.processing_completed)
            self.processing_thread.error_signal.connect(self.processing_error)

        # Start processing
        self.processing_thread.start()
    
    def processing_completed(self):
        """Handle successful processing completion."""
        self.window.show_progress_bar(False)
        
        # Build success message
        message = f"Output files have been successfully generated:\n"
        message += f"Video: {os.path.basename(self.window.output_video_path)}\n"
        message += f"CSV 1: {os.path.basename(self.window.output_csv_path)}\n"
        
        if self.window.output_second_csv_path:
            message += f"CSV 2: {os.path.basename(self.window.output_second_csv_path)}\n"
        
        QMessageBox.information(
            self.window,
            "Success",
            message
        )
    
    def processing_error(self, error_message):
        """Handle processing error."""
        self.window.show_progress_bar(False)
        QMessageBox.critical(self.window, "Error", error_message)
    
    def scan_batch_directory(self, directory):
        """Scan a directory for matching video and CSV files."""
        # Use batch processor to find matching files
        file_sets = self.batch_processor.find_matching_files(directory)
        
        # Update the UI with the matched files
        self.window.update_files_table(file_sets)
        
        # Set these file sets in the batch processor
        self.batch_processor.set_file_sets(file_sets)
        
        # Show message if no matches found
        if not file_sets:
            QMessageBox.warning(
                self.window,
                "No Matches Found",
                "No matching video and CSV file sets were found in the selected directory."
            )
    
    def load_batch_file_set(self, file_set):
        """
        Load a specific file set into the annotation interface.
        
        Args:
            file_set: Dictionary with video and CSV paths
        
        Returns:
            True if successful, False otherwise
        """
        if not file_set:
            return False
            
        # Pause any current video playback
        self.video_player.pause()
        
        # Clear any current actions
        self.action_tracker.clear_actions()
        self.active_action = None
        self.window.current_active_action = None
        
        # Set file paths in window
        self.window.video_path = file_set['video']
        self.window.video_path_label.setText(os.path.basename(file_set['video']))
        
        self.window.csv_path = file_set['csv1']
        if file_set['csv1']:
            self.window.csv_path_label.setText(os.path.basename(file_set['csv1']))
        else:
            self.window.csv_path_label.setText("No CSV 1 selected")
            
        self.window.second_csv_path = file_set['csv2']
        if file_set['csv2']:
            self.window.second_csv_path_label.setText(os.path.basename(file_set['csv2']))
        else:
            self.window.second_csv_path_label.setText("No CSV 2 selected")
        
        # Initialize video player and CSV handler
        success_video = self.init_video_player()
        success_csv = self.init_csv_handler()
        
        return success_video and success_csv
    
    def load_first_file(self):
        """Load the first file in the batch."""
        # Get the first file set
        file_set = self.batch_processor.load_first_file()
        if not file_set:
            QMessageBox.warning(
                self.window,
                "No Files",
                "No files available in the batch."
            )
            return
        
        # Load the file set
        if self.load_batch_file_set(file_set):
            # Update batch mode UI
            total_files = self.batch_processor.get_total_files()
            current_index = self.batch_processor.get_current_index()
            self.window.set_batch_mode(True, current_index, total_files)
    
    def load_next_file(self):
        """Load the next file in the batch."""
        # Check if there's a next file
        if not self.batch_processor.has_next_file():
            QMessageBox.information(
                self.window,
                "End of Batch",
                "You have reached the last file in the batch."
            )
            return
        
        # Get the next file set
        file_set = self.batch_processor.load_next_file()
        if not file_set:
            return
        
        # Load the file set
        if self.load_batch_file_set(file_set):
            # Update batch mode UI
            total_files = self.batch_processor.get_total_files()
            current_index = self.batch_processor.get_current_index()
            self.window.set_batch_mode(True, current_index, total_files)
    
    def load_previous_file(self):
        """Load the previous file in the batch."""
        # Check if there's a previous file
        if not self.batch_processor.has_previous_file():
            QMessageBox.information(
                self.window,
                "Start of Batch",
                "You are already at the first file in the batch."
            )
            return
        
        # Get the previous file set
        file_set = self.batch_processor.load_previous_file()
        if not file_set:
            return
        
        # Load the file set
        if self.load_batch_file_set(file_set):
            # Update batch mode UI
            total_files = self.batch_processor.get_total_files()
            current_index = self.batch_processor.get_current_index()
            self.window.set_batch_mode(True, current_index, total_files)

    def batch_file_processed(self):
        """Handle completion of processing a file in batch mode."""
        self.window.show_progress_bar(False)

        # Check if there are more files to process
        if self.batch_processor.has_next_file():
            # Automatically continue to the next file without asking
            self.load_next_file()
        else:
            # This was the last file, show completion message
            QMessageBox.information(
                self.window,
                "Batch Processing Complete",
                "All files in the batch have been processed!"
            )

            # Reset batch mode
            self.window.set_batch_mode(False)
    
    def batch_processing_error(self, error_message):
        """Handle error during batch processing."""
        self.window.show_progress_bar(False)
        
        # Show error message with option to continue
        reply = QMessageBox.warning(
            self.window,
            "Processing Error",
            f"Error processing file: {error_message}\n\nDo you want to continue to the next file?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes and self.batch_processor.has_next_file():
            # Continue to next file
            self.load_next_file()
        else:
            # Exit batch mode
            QMessageBox.critical(
                self.window,
                "Batch Processing Stopped",
                "Batch processing has been stopped due to an error."
            )
    
    def run(self):
        """Run the application."""
        return self.app.exec_()


if __name__ == "__main__":
    add_exception_logging()
    controller = ApplicationController()
    sys.exit(controller.run())