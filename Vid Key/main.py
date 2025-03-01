# main.py - Main application entry point with enhanced video player
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
        """Run the processing tasks."""
        try:
            # Add actions to CSV(s)
            self.progress_signal.emit(10)
            if not self.csv_handler.add_action_column(self.action_tracker.get_all_actions()):
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
        Event filter to prioritize key events.
        This helps make key presses more responsive by processing them
        outside the normal event loop.
        """
        if obj == self.window:
            if event.type() == QEvent.KeyPress:
                # For key events, prioritize handling
                if hasattr(event, 'text') and event.text() in self.window.key_to_action:
                    # Process immediately without waiting for normal event queue
                    self.window.keyPressEvent(event)
                    return True  # Indicate we've handled this event

            elif event.type() == QEvent.KeyRelease:
                if hasattr(event, 'text') and event.text() in self.window.key_to_action:
                    self.window.keyReleaseEvent(event)
                    return True

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
            
            # Check if UI state needs updating
            if self.window.current_active_action != self.active_action:
                # Update window state
                self.window.current_active_action = self.active_action
                
                # Ensure button visual state is correct
                if self.active_action in self.window.action_buttons:
                    button = self.window.action_buttons[self.active_action]
                    if not button.pressed_state:
                        button.set_pressed(True)
                        
                # Make sure other buttons are not pressed
                for code, button in self.window.action_buttons.items():
                    if code != self.active_action and button.pressed_state:
                        button.set_pressed(False)
            
            # Check if we need to update the action tracker
            if current_frame != self.action_tracker.current_frame:
                self.action_tracker.set_frame(current_frame)
                self.action_tracker.continue_action()

    def connect_signals(self):
        """Connect all signals and slots."""
        # Main window signals
        self.window.play_pause_signal.connect(self.toggle_play_pause)
        self.window.frame_changed_signal.connect(self.seek_to_frame)
        self.window.save_signal.connect(self.save_outputs)
        self.window.action_started_signal.connect(self.start_action)
        self.window.action_stopped_signal.connect(self.stop_action)
        self.window.action_continued_signal.connect(self.continue_action)

        # Video player signals
        self.video_player.frameChanged.connect(self.update_frame)
        self.video_player.videoFinished.connect(self.video_finished)

    def continue_action(self):
        """Continue the current action for the current frame."""
        try:
            if self.active_action:
                # Update action tracker with current frame
                current_frame = self.video_player.current_frame
                self.action_tracker.set_frame(current_frame)
                self.action_tracker.continue_action()
        except Exception as e:
            print(f"Error in continue_action: {str(e)}")

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
            self.action_tracker.continue_action()
            
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
        if not self.window.video_path:
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
            # Force continue the active action for this frame
            self.action_tracker.continue_action()
            display_action = self.active_action

            # Ensure button stays visually pressed
            if self.active_action in self.window.action_buttons:
                self.window.action_buttons[self.active_action].set_pressed(True)

        # Priority 2: Previously recorded action for this frame
        elif not display_action:
            # Get any previously coded action for this frame
            previous_action = self.action_tracker.get_action_for_frame(frame_number)
            if previous_action:
                display_action = previous_action

        # Update the video player's overlay - only if needed (different from current)
        if display_action and display_action != self.video_player.current_action:
            self.video_player.set_action(display_action)
        elif not display_action and self.video_player.current_action:
            self.video_player.clear_action()

    def toggle_play_pause(self, play):
        """Toggle between play and pause states."""
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
            # First set the active action - do this FIRST for responsiveness
            self.active_action = action_code
            self.window.current_active_action = action_code
            
            # Update action tracker with current frame
            current_frame = self.video_player.current_frame
            self.action_tracker.set_frame(current_frame)
            self.action_tracker.start_action(action_code)
            
            # Update overlay display - but don't do heavy redrawing
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
        # Clear active action immediately for responsiveness
        current_action = self.active_action
        self.active_action = None
        self.window.current_active_action = None

        try:
            # Update action tracker
            self.action_tracker.stop_action()

            # Clear action in video player only if an action is currently set
            if hasattr(self.video_player, 'current_action') and self.video_player.current_action:
                self.video_player.clear_action()
                
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
            
        # Check CSV loaded
        if not self.csv_handler.data and not self.init_csv_handler():
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
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
        if not self.action_tracker.get_all_actions():
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
    
    def run(self):
        """Run the application."""
        return self.app.exec_()


if __name__ == "__main__":
    add_exception_logging()
    controller = ApplicationController()
    sys.exit(controller.run())
