# gui_component.py - Creates the user interface
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QSlider, QFileDialog, QGroupBox,
                           QProgressBar, QMessageBox, QSplitter, QSizePolicy, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont
import config
import os

class ActionButton(QPushButton):
    """Custom button for press-and-hold action activation."""
    pressed_signal = pyqtSignal(str)
    released_signal = pyqtSignal()

    def __init__(self, action_code, action_text, key_shortcut, parent=None):
        """Initialize the action button."""
        super().__init__(action_text, parent)
        self.action_code = action_code
        self.setMinimumHeight(40)
        self.setFont(QFont('Arial', 10, QFont.Bold))  # Slightly larger font

        # Use QSS stylesheets instead of programmatic styling - much faster
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {config.BUTTON_COLORS['normal']};
                color: {config.BUTTON_COLORS['text_normal']};
                border: 1px solid #999999;
                border-radius: 5px;
                padding: 5px;
            }}
            QPushButton:pressed {{
                background-color: {config.BUTTON_COLORS['pressed']};
                color: {config.BUTTON_COLORS['text_pressed']};
                border: 1px solid #4a6ea9;
            }}
        """)

        self.pressed_state = False

        # Connect press and release signals for press-and-hold behavior
        self.pressed.connect(self.on_pressed)
        self.released.connect(self.on_released)

        # Store shortcut key for display/reference only, but DON'T set active shortcut
        self.key_shortcut = key_shortcut
        # No setShortcut() call to prevent automatic button activation

    def on_pressed(self):
        """Handle button press event."""
        # Emit signal with action code for parent to handle
        self.pressed_signal.emit(self.action_code)
        # Ensure button gets proper visual focus
        self.setFocus()

    def on_released(self):
        """Handle button release event."""
        # Emit release signal for parent to handle
        self.released_signal.emit()

    def set_pressed(self, pressed=True):
        """Set the button's pressed state programmatically."""
        self.pressed_state = pressed
        if pressed:
            # Use style property instead of full stylesheet replacement
            self.setProperty("pressed", True)
            self.setDown(True)  # Makes button appear pressed
        else:
            self.setProperty("pressed", False)
            self.setDown(False)  # Makes button appear normal

        # Force style refresh - more efficient than setting full stylesheet
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

class MainWindow(QMainWindow):
    """Main application window."""
    frame_changed_signal = pyqtSignal(int)
    play_pause_signal = pyqtSignal(bool)  # True for play, False for pause
    save_signal = pyqtSignal()
    progress_update_signal = pyqtSignal(int)  # Progress percentage
    action_started_signal = pyqtSignal(str)  # Action code
    action_stopped_signal = pyqtSignal()  # No action
    action_continued_signal = pyqtSignal()  # Signal for continuing the current action

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Facial Action Coder")
        # Main layout adjustments
        self.setMinimumSize(1200, 800)
        
        # Dictionary to track key states and active action
        self.pressed_keys = {}
        self.key_to_action = {}  # Maps key text to action code
        self.current_active_action = None  # Currently active action code
        
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        # Apply standardized styles for the entire application
        self.setStyleSheet(config.GROUP_BOX_STYLE)

        # Main widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)  # Horizontal layout
        main_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                       config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        main_layout.setSpacing(config.STANDARD_SPACING)

        # Left section: Video display and controls
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(config.STANDARD_SPACING)

        video_section = QGroupBox("Video Player")
        font = QFont(*config.SECTION_TITLE_FONT)
        font.setWeight(config.SECTION_TITLE_WEIGHT)
        video_section.setFont(font)

        video_inner_layout = QVBoxLayout(video_section)
        video_inner_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                              config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        video_inner_layout.setSpacing(config.STANDARD_SPACING)

        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(768, 680)
        self.video_label.setStyleSheet("background-color: black")

        video_inner_layout.addWidget(self.video_label, 1)

        # Video controls
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                           config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        controls_layout.setSpacing(config.STANDARD_SPACING)

        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.setFixedWidth(60)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(10)
        self.frame_slider.valueChanged.connect(self.slider_changed)

        self.frame_label = QLabel("Frame: 0/0")
        self.frame_label.setFixedWidth(100)
        self.frame_label.setFont(QFont('Arial', 9))

        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.frame_label)

        video_inner_layout.addLayout(controls_layout)
        video_layout.addWidget(video_section, 1)

        # Right section: File selection and action buttons
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                        config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        right_layout.setSpacing(config.STANDARD_SPACING * 2)  # Slightly larger spacing between sections

        # File selection section
        file_section = QGroupBox("Input/Output Files")
        font = QFont(*config.SECTION_TITLE_FONT)
        font.setWeight(config.SECTION_TITLE_WEIGHT)
        file_section.setFont(font)

        file_layout = QVBoxLayout(file_section)
        file_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                       config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        file_layout.setSpacing(config.STANDARD_SPACING)

        # Video selection
        video_file_layout = QHBoxLayout()
        video_file_layout.setSpacing(config.STANDARD_SPACING)

        select_video_btn = QPushButton("Select Video")
        select_video_btn.setFixedWidth(100)
        select_video_btn.clicked.connect(self.select_video)
        select_video_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE)

        self.video_path_label = QLabel("No video selected")
        self.video_path_label.setTextFormat(Qt.PlainText)
        self.video_path_label.setWordWrap(True)
        self.video_path_label.setFont(QFont('Arial', 9))

        video_file_layout.addWidget(select_video_btn)
        video_file_layout.addWidget(self.video_path_label, 1)

        # CSV selection
        csv_file_layout = QHBoxLayout()
        csv_file_layout.setSpacing(config.STANDARD_SPACING)

        select_csv_btn = QPushButton("Select CSV")
        select_csv_btn.setFixedWidth(100)
        select_csv_btn.clicked.connect(self.select_csv_files)
        select_csv_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE)

        csv_file_layout.addWidget(select_csv_btn)

        # CSV file status
        csv_status_layout = QVBoxLayout()
        csv_status_layout.setSpacing(config.STANDARD_SPACING)

        self.csv_path_label = QLabel("No CSV 1 selected")
        self.csv_path_label.setTextFormat(Qt.PlainText)
        self.csv_path_label.setWordWrap(True)
        self.csv_path_label.setFont(QFont('Arial', 9))

        self.second_csv_path_label = QLabel("No CSV 2 selected")
        self.second_csv_path_label.setTextFormat(Qt.PlainText)
        self.second_csv_path_label.setWordWrap(True)
        self.second_csv_path_label.setFont(QFont('Arial', 9))

        csv_status_layout.addWidget(self.csv_path_label)
        csv_status_layout.addWidget(self.second_csv_path_label)

        csv_file_layout.addLayout(csv_status_layout)

        file_layout.addLayout(video_file_layout)
        file_layout.addLayout(csv_file_layout)

        right_layout.addWidget(file_section)

        # Current Action section (converted to QGroupBox for consistency)
        self.action_display_section = QGroupBox("Current Action")
        font = QFont(*config.SECTION_TITLE_FONT)
        font.setWeight(config.SECTION_TITLE_WEIGHT)
        self.action_display_section.setFont(font)

        action_display_layout = QVBoxLayout(self.action_display_section)
        action_display_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                                 config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        action_display_layout.setSpacing(config.STANDARD_SPACING)

        self.action_display_label = QLabel("No action")
        self.action_display_label.setAlignment(Qt.AlignCenter)
        self.action_display_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.action_display_label.setStyleSheet(f"""
            background-color: {config.UI_COLORS['highlight']};
            color: white;
            border-radius: 3px;
            padding: 6px;
            min-height: 35px;
        """)

        action_display_layout.addWidget(self.action_display_label)

        right_layout.addWidget(self.action_display_section)

        # Action buttons section
        action_section = QGroupBox("Action Buttons (Press and Hold)")
        font = QFont(*config.SECTION_TITLE_FONT)
        font.setWeight(config.SECTION_TITLE_WEIGHT)
        action_section.setFont(font)

        action_inner_layout = QVBoxLayout(action_section)
        action_inner_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                               config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        action_inner_layout.setSpacing(config.STANDARD_SPACING)

        # Create action buttons
        self.action_buttons = {}

        # Define our desired order and key mappings
        button_order = [
            ("RE", "1"),  # Raise Eyebrows - key 1
            ("ES", "2"),  # Close Eyes Softly - key 2
            ("ET", "3"),  # Close Eyes Tightly - key 3
            ("SS", "4"),  # Soft Smile - key 4
            ("BS", "5"),  # Big Smile - key 5
            ("SO", "6"),  # Say 'O' - key 6
            ("SE", "7"),  # Say 'E' - key 7
            ("BL", "8"),  # Blink - key 8
            ("WN", "9"),  # Wrinkle Nose - key 9
            ("PL", "0")  # Pucker Lips - key 0
        ]

        # Create buttons in our specified order with assigned keys
        for code, key_shortcut in button_order:
            # Get action text from config
            text = config.ACTION_MAPPINGS.get(code, "Unknown Action")

            # Format button text to show action description and key shortcut
            button_text = f"{text} ({key_shortcut})"

            # Create the button
            button = ActionButton(code, button_text, key_shortcut)
            button.pressed_signal.connect(self.action_pressed)
            button.released_signal.connect(self.action_released)

            # Add to layout
            action_inner_layout.addWidget(button)

            # Store in our button dictionary
            self.action_buttons[code] = button

            # Map key to action for keyboard handling
            self.key_to_action[key_shortcut] = code

        right_layout.addWidget(action_section)

        # Progress bar and generate button
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(config.STANDARD_SPACING)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        save_btn = QPushButton("Generate Output Files")
        save_btn.clicked.connect(self.save_outputs)
        save_btn.setMinimumHeight(35)
        save_btn.setFont(QFont('Arial', 9, QFont.Bold))
        save_btn.setStyleSheet(config.PRIMARY_BUTTON_STYLE)

        bottom_layout.addWidget(self.progress_bar)
        bottom_layout.addWidget(save_btn)

        right_layout.addLayout(bottom_layout)

        # Add the main horizontal sections with adjusted proportions
        main_layout.addWidget(video_widget, 2)
        main_layout.addWidget(right_panel, 1)

        self.setCentralWidget(central_widget)

        # Initialize file path variables
        self.video_path = None
        self.csv_path = None
        self.second_csv_path = None
        self.output_csv_path = None
        self.output_second_csv_path = None
        self.output_video_path = None

        # Connect the progress signal
        self.progress_update_signal.connect(self.update_progress)

        # Set up application-wide event handling priorities
        if QApplication.instance():
            QApplication.instance().setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

        # Ensure focus when window is shown
        self.activateWindow()

    def update_action_display(self, action_code):
        """Update the action display with the current action."""
        if not action_code:
            self.action_display_label.setText("No action")
            self.action_display_label.setStyleSheet(f"""
                background-color: {config.UI_COLORS['section_bg']};
                color: {config.UI_COLORS['text_inactive']};
                border-radius: 3px;
                padding: 6px;
                min-height: 35px;
            """)
            return

        # Get action text from config
        action_text = config.ACTION_MAPPINGS.get(action_code, "Unknown Action")
        display_text = f"{action_code}: {action_text}"

        # Set text and style based on action
        self.action_display_label.setText(display_text)
        self.action_display_label.setStyleSheet(f"""
            background-color: {config.UI_COLORS['highlight']};
            color: white;
            border-radius: 3px;
            padding: 6px;
            min-height: 35px;
            font-weight: bold;
        """)

    def select_video(self):
        """Open file dialog to select input video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.video_path = file_path
            self.video_path_label.setText(os.path.basename(file_path))
    
    def select_csv_files(self):
        """Open file dialog to select one or two CSV files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV File(s)", "", "CSV Files (*.csv)"
        )
        
        if file_paths:
            # Clear previous selections
            self.csv_path = None
            self.second_csv_path = None
            self.csv_path_label.setText("No CSV 1 selected")
            self.second_csv_path_label.setText("No CSV 2 selected")
            
            # Set the first file
            if len(file_paths) >= 1:
                self.csv_path = file_paths[0]
                self.csv_path_label.setText(os.path.basename(file_paths[0]))
            
            # Set the second file if available
            if len(file_paths) >= 2:
                self.second_csv_path = file_paths[1]
                self.second_csv_path_label.setText(os.path.basename(file_paths[1]))
                
            # Warn if more than 2 files were selected
            if len(file_paths) > 2:
                QMessageBox.warning(
                    self, 
                    "Too Many Files", 
                    "More than 2 CSV files were selected. Only the first 2 will be used."
                )
    
    def update_video_frame(self, frame_number, qimage, action_code):
        """Update the video display with a new frame."""
        pixmap = QPixmap.fromImage(qimage)
        
        # Scale maintaining aspect ratio, but optimized for the new dimensions
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), 
            self.video_label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
    
    def update_frame_info(self, current_frame, total_frames):
        """Update the frame information display."""
        self.frame_label.setText(f"Frame: {current_frame}/{total_frames}")
        # Update slider without triggering valueChanged signal
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(current_frame)
        self.frame_slider.blockSignals(False)

    def set_total_frames(self, total_frames):
        """Set the total number of frames for the slider."""
        # Ensure we have at least one frame
        total_frames = max(1, total_frames)

        # Set maximum to total frames - 1 (zero-based indexing)
        self.frame_slider.setMaximum(total_frames - 1)

        # Update range display in frame label
        current_frame = self.frame_slider.value()
        self.frame_label.setText(f"Frame: {current_frame}/{total_frames - 1}")
    
    def toggle_play_pause(self):
        """Toggle between play and pause states."""
        if self.play_pause_btn.text() == "Play":
            self.play_pause_btn.setText("Pause")
            self.play_pause_signal.emit(True)
        else:
            self.play_pause_btn.setText("Play")
            self.play_pause_signal.emit(False)
    
    def slider_changed(self, value):
        """Handle slider value changes."""
        self.frame_changed_signal.emit(value)

    def action_pressed(self, action_code):
        """Handle action button press event."""
        # Clear all other buttons first to ensure only one is active
        self.clear_all_action_buttons()

        # Set as current active action
        self.current_active_action = action_code

        # Highlight the correct button
        if action_code in self.action_buttons:
            self.action_buttons[action_code].set_pressed(True)

        # Emit signal for action started
        self.action_started_signal.emit(action_code)
    
    def action_released(self):
        """Handle action button release event."""
        # Clear the active action
        self.current_active_action = None
        
        # Clear all buttons
        self.clear_all_action_buttons()
        
        # Emit signal for action stopped
        self.action_stopped_signal.emit()

    def clear_all_action_buttons(self):
        """Clear the pressed state of all action buttons."""
        for code, button in self.action_buttons.items():
            if code != self.current_active_action:
                button.set_pressed(False)
            else:
                button.set_pressed(True)  # Ensure active button stays pressed
    
    def update_progress(self, value):
        """Update progress bar value."""
        self.progress_bar.setValue(value)
    
    def show_progress_bar(self, visible=True):
        """Show or hide the progress bar."""
        self.progress_bar.setVisible(visible)
        if visible:
            self.progress_bar.setValue(0)
    
    def save_outputs(self):
        """Trigger the generation of output files."""
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please select an input video file first.")
            return
            
        if not self.csv_path:
            QMessageBox.warning(self, "Warning", "Please select at least one input CSV file first.")
            return
            
        self.save_signal.emit()
        self.show_progress_bar(True)

    def keyPressEvent(self, event):
        """
        Handle keyboard press events.
        """
        try:
            key = event.text()

            # Check if key is in our action mapping
            if key in self.key_to_action:
                action_code = self.key_to_action[key]
                
                # Only process this key if it's not already being tracked as pressed
                if not self.pressed_keys.get(key, False):
                    # Record that this key is pressed
                    self.pressed_keys[key] = True
                    
                    # Set as current active action
                    self.current_active_action = action_code

                    # Highlight the correct button
                    if action_code in self.action_buttons:
                        self.action_buttons[action_code].set_pressed(True)

                    # Emit signal for action started
                    self.action_started_signal.emit(action_code)

                # Always accept the event
                event.accept()
                return
        except Exception as e:
            print(f"Error in keyPressEvent: {str(e)}")

        # Pass unhandled events to parent
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """
        Handle keyboard release events.
        """
        try:
            key = event.text()

            # Only process if this key was previously recorded as pressed
            if key in self.key_to_action and self.pressed_keys.get(key, False):
                action_code = self.key_to_action[key]

                # Clear the pressed state
                self.pressed_keys[key] = False
                
                # Only deactivate if this matches the current active action
                if action_code == self.current_active_action:
                    self.current_active_action = None

                    # Reset button appearance
                    if action_code in self.action_buttons:
                        self.action_buttons[action_code].set_pressed(False)

                    # Emit the release signal
                    self.action_stopped_signal.emit()

                # Accept the event
                event.accept()
                return
        except Exception as e:
            print(f"Error in keyReleaseEvent: {str(e)}")
        
        # Pass unhandled events to parent for normal processing
        super().keyReleaseEvent(event)
