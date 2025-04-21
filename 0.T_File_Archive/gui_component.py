# gui_component.py - Creates the user interface with sequential batch processing
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QSlider, QFileDialog, QGroupBox,
                           QProgressBar, QMessageBox, QSplitter, QSizePolicy, QApplication,
                           QTabWidget, QListWidget, QListWidgetItem, QRadioButton, 
                           QButtonGroup, QCheckBox, QTableWidget, QTableWidgetItem,
                           QHeaderView, QAbstractItemView)
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
    """Main application window with sequential batch processing support."""
    # Original signals
    frame_changed_signal = pyqtSignal(int)
    play_pause_signal = pyqtSignal(bool)  # True for play, False for pause
    save_signal = pyqtSignal()
    progress_update_signal = pyqtSignal(int)  # Progress percentage
    action_started_signal = pyqtSignal(str)  # Action code
    action_stopped_signal = pyqtSignal()  # No action
    action_continued_signal = pyqtSignal()  # Signal for continuing the current action
    
    # Batch mode signals
    select_batch_dir_signal = pyqtSignal(str)  # Directory path
    next_file_signal = pyqtSignal()
    previous_file_signal = pyqtSignal()
    first_file_signal = pyqtSignal()
    
    # New signal for multi-video selection
    videos_selected_signal = pyqtSignal(list)  # List of video file paths

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
        
        # Initialize file path variables
        self.video_path = None
        self.csv_path = None
        self.second_csv_path = None
        self.output_csv_path = None
        self.output_second_csv_path = None
        self.output_video_path = None
        self.batch_directory = None
        self.batch_mode = False  # Flag to indicate if in batch mode
        
        # Initialize UI elements that will be accessed across methods
        self.action_buttons = {}
        self.video_label = None
        self.play_pause_btn = None
        self.frame_slider = None
        self.frame_label = None
        self.select_video_btn = None
        self.select_csv_btn = None
        self.video_path_label = None
        self.csv_path_label = None
        self.second_csv_path_label = None
        self.action_display_label = None
        self.progress_bar = None
        self.save_btn = None
        self.batch_status_label = None
        self.prev_file_btn = None
        self.next_file_btn = None
        self.batch_dir_label = None
        self.files_table = None
        self.start_batch_btn = None
        self.batch_nav_layout = None
        
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        # Apply standardized styles for the entire application
        self.setStyleSheet(config.GROUP_BOX_STYLE)

        # Main widget and tab layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                      config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        
        # Create tab widget for manual vs batch mode
        self.tab_widget = QTabWidget()
        
        # Single file mode tab (original functionality)
        self.single_file_tab = QWidget()
        self.setup_single_file_tab()
        self.tab_widget.addTab(self.single_file_tab, "Manual Annotation")
        
        # Batch mode tab (file selection functionality)
        self.batch_tab = QWidget()
        self.setup_batch_tab()
        self.tab_widget.addTab(self.batch_tab, "Batch Processing")
        
        # Tab switch signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        main_layout.addWidget(self.tab_widget)
        
        self.setCentralWidget(central_widget)

        # Connect the progress signal
        self.progress_update_signal.connect(self.update_progress)

        # Now that all UI elements are created, we can safely set the batch mode
        # This must happen AFTER the save_btn has been created
        self.set_batch_mode(False)
        
        # Set up application-wide event handling priorities
        if QApplication.instance():
            QApplication.instance().setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

        # Ensure focus when window is shown
        self.activateWindow()

    def setup_single_file_tab(self):
        """Set up the single file annotation tab (original functionality)."""
        main_layout = QHBoxLayout(self.single_file_tab)
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

        self.select_video_btn = QPushButton("Select Videos")  # Changed from "Select Video"
        self.select_video_btn.setFixedWidth(100)
        self.select_video_btn.clicked.connect(self.select_video)
        self.select_video_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE)

        self.video_path_label = QLabel("No video selected")
        self.video_path_label.setTextFormat(Qt.PlainText)
        self.video_path_label.setWordWrap(True)
        self.video_path_label.setFont(QFont('Arial', 9))

        video_file_layout.addWidget(self.select_video_btn)
        video_file_layout.addWidget(self.video_path_label, 1)

        # CSV file status
        csv_file_layout = QVBoxLayout()
        csv_file_layout.setSpacing(config.STANDARD_SPACING)

        csv_label = QLabel("Matched CSV Files:")
        csv_label.setFont(QFont('Arial', 9, QFont.Bold))
        
        self.csv_path_label = QLabel("No CSV 1 selected")
        self.csv_path_label.setTextFormat(Qt.PlainText)
        self.csv_path_label.setWordWrap(True)
        self.csv_path_label.setFont(QFont('Arial', 9))

        self.second_csv_path_label = QLabel("No CSV 2 selected")
        self.second_csv_path_label.setTextFormat(Qt.PlainText)
        self.second_csv_path_label.setWordWrap(True)
        self.second_csv_path_label.setFont(QFont('Arial', 9))

        csv_file_layout.addWidget(csv_label)
        csv_file_layout.addWidget(self.csv_path_label)
        csv_file_layout.addWidget(self.second_csv_path_label)

        file_layout.addLayout(video_file_layout)
        file_layout.addLayout(csv_file_layout)

        # Batch navigation controls - only visible in batch mode
        self.batch_nav_layout = QHBoxLayout()
        self.batch_nav_layout.setSpacing(config.STANDARD_SPACING)
        
        self.batch_status_label = QLabel("Batch Mode: File 0 of 0")
        self.batch_status_label.setAlignment(Qt.AlignCenter)
        self.batch_status_label.setFont(QFont('Arial', 9, QFont.Bold))
        
        self.prev_file_btn = QPushButton("← Previous")
        self.prev_file_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE)
        self.prev_file_btn.clicked.connect(self.load_previous_file)
        self.prev_file_btn.setEnabled(False)
        
        self.next_file_btn = QPushButton("Next →")
        self.next_file_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE)
        self.next_file_btn.clicked.connect(self.load_next_file)
        self.next_file_btn.setEnabled(False)
        
        self.batch_nav_layout.addWidget(self.prev_file_btn)
        self.batch_nav_layout.addWidget(self.batch_status_label, 1)
        self.batch_nav_layout.addWidget(self.next_file_btn)
        
        # Add batch navigation layout
        file_layout.addLayout(self.batch_nav_layout)
        
        # Add file section to right layout
        right_layout.addWidget(file_section)

        # Current Action section (converted to QGroupBox for consistency)
        action_display_section = QGroupBox("Current Action")
        font = QFont(*config.SECTION_TITLE_FONT)
        font.setWeight(config.SECTION_TITLE_WEIGHT)
        action_display_section.setFont(font)

        action_display_layout = QVBoxLayout(action_display_section)
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

        right_layout.addWidget(action_display_section)

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

        # Define our desired order and key mappings - UPDATED
        button_order = [
            ("RE", "1"),  # Raise Eyebrows - key 1
            ("ES", "2"),  # Close Eyes Softly - key 2
            ("ET", "3"),  # Close Eyes Tightly - key 3
            ("SS", "4"),  # Soft Smile - key 4
            ("BS", "5"),  # Big Smile - key 5
            ("SO", "6"),  # Say 'O' - key 6
            ("SE", "7"),  # Say 'E' - key 7
            ("BL", "0")   # Baseline - key 0
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

        # Create save button as a class member
        self.save_btn = QPushButton("Generate Output Files")
        self.save_btn.clicked.connect(self.save_outputs)
        self.save_btn.setMinimumHeight(35)
        self.save_btn.setFont(QFont('Arial', 9, QFont.Bold))
        self.save_btn.setStyleSheet(config.PRIMARY_BUTTON_STYLE)

        bottom_layout.addWidget(self.progress_bar)
        bottom_layout.addWidget(self.save_btn)

        right_layout.addLayout(bottom_layout)

        # Add the main horizontal sections with adjusted proportions
        main_layout.addWidget(video_widget, 2)
        main_layout.addWidget(right_panel, 1)

    def setup_batch_tab(self):
        """Set up the batch processing tab for file selection."""
        main_layout = QVBoxLayout(self.batch_tab)
        main_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                     config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        main_layout.setSpacing(config.STANDARD_SPACING * 2)
        
        # Directory selection
        dir_section = QGroupBox("Batch Directory")
        font = QFont(*config.SECTION_TITLE_FONT)
        font.setWeight(config.SECTION_TITLE_WEIGHT)
        dir_section.setFont(font)
        
        dir_layout = QVBoxLayout(dir_section)
        dir_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                    config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        dir_layout.setSpacing(config.STANDARD_SPACING)
        
        dir_select_layout = QHBoxLayout()
        dir_select_layout.setSpacing(config.STANDARD_SPACING)
        
        select_dir_btn = QPushButton("Select Directory")
        select_dir_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE)
        select_dir_btn.clicked.connect(self.select_batch_directory)
        
        self.batch_dir_label = QLabel("No directory selected")
        self.batch_dir_label.setTextFormat(Qt.PlainText)
        self.batch_dir_label.setWordWrap(True)
        self.batch_dir_label.setFont(QFont('Arial', 9))
        
        dir_select_layout.addWidget(select_dir_btn)
        dir_select_layout.addWidget(self.batch_dir_label, 1)
        
        dir_layout.addLayout(dir_select_layout)
        
        main_layout.addWidget(dir_section)
        
        # Matched files table
        files_section = QGroupBox("Matched Files")
        font = QFont(*config.SECTION_TITLE_FONT)
        font.setWeight(config.SECTION_TITLE_WEIGHT)
        files_section.setFont(font)
        
        files_layout = QVBoxLayout(files_section)
        files_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN,
                                      config.STANDARD_MARGIN, config.STANDARD_MARGIN)
        files_layout.setSpacing(config.STANDARD_SPACING)
        
        self.files_table = QTableWidget(0, 4)  # Start with 0 rows, 4 columns
        self.files_table.setHorizontalHeaderLabels(["Base ID", "Video", "CSV 1", "CSV 2"])
        self.files_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.files_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.files_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.files_table.doubleClicked.connect(self.table_double_clicked)
        
        files_layout.addWidget(self.files_table)
        
        # Start button for batch
        start_layout = QHBoxLayout()
        
        self.start_batch_btn = QPushButton("Start Processing First File")
        self.start_batch_btn.setStyleSheet(config.PRIMARY_BUTTON_STYLE)
        self.start_batch_btn.setMinimumHeight(35)
        self.start_batch_btn.setFont(QFont('Arial', 9, QFont.Bold))
        self.start_batch_btn.clicked.connect(self.start_batch_processing)
        self.start_batch_btn.setEnabled(False)
        
        start_layout.addStretch()
        start_layout.addWidget(self.start_batch_btn)
        start_layout.addStretch()
        
        files_layout.addLayout(start_layout)
        
        main_layout.addWidget(files_section)

    def set_batch_mode(self, enabled, current_index=-1, total_files=0):
        """
        Enable or disable batch mode UI elements.
        
        Args:
            enabled: True to show batch controls, False to hide
            current_index: Current file index (0-based)
            total_files: Total number of files in batch
        """
        self.batch_mode = enabled
        
        # Show/hide batch navigation controls
        for widget in [self.batch_status_label, self.prev_file_btn, self.next_file_btn]:
            widget.setVisible(enabled)
        
        # Show/hide manual selection buttons
        self.select_video_btn.setEnabled(not enabled)
        
        # Update batch status label if enabled
        if enabled:
            # Update display (add 1 to index for 1-based display)
            self.batch_status_label.setText(f"Batch Mode: File {current_index + 1} of {total_files}")
            
            # Enable/disable navigation buttons based on position
            self.prev_file_btn.setEnabled(current_index > 0)
            self.next_file_btn.setEnabled(current_index < total_files - 1)
            
            # Update the save button text to indicate continuation to next file
            if current_index < total_files - 1:
                self.save_btn.setText("Save and Continue to Next File")
            else:
                self.save_btn.setText("Save and Complete Batch")
        else:
            # Reset to standard text when not in batch mode
            self.save_btn.setText("Generate Output Files")

    def on_tab_changed(self, index):
        """Handle tab change events."""
        # If switching to batch tab, we'll just update the UI without emitting signals
        if index == 1:  # Batch tab
            self.play_pause_btn.setText("Play")
            # Don't emit the signal as it causes the "Please select video" warning
            # self.play_pause_signal.emit(False)

            # If we have direct access to the video player, pause it directly
            if hasattr(self, 'video_player') and self.video_player:
                self.video_player.pause()
    
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
        """Open file dialog to select input video(s)."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video File(s)", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_paths:
            # Emit signal with the selected video paths
            self.videos_selected_signal.emit(file_paths)
    

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

    def select_batch_directory(self):
        """Open directory dialog to select batch processing directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory Containing Video and CSV Files"
        )
        
        if directory:
            self.batch_directory = directory
            self.batch_dir_label.setText(directory)
            # Emit signal to scan directory
            self.select_batch_dir_signal.emit(directory)
    
    def update_files_table(self, file_sets):
        """Update the matched files table with file sets found in the directory."""
        # Clear the table
        self.files_table.setRowCount(0)
        
        # Add rows for each file set
        for i, file_set in enumerate(file_sets):
            self.files_table.insertRow(i)
            
            # Add cells with file information
            self.files_table.setItem(i, 0, QTableWidgetItem(file_set['base_id']))
            self.files_table.setItem(i, 1, QTableWidgetItem(os.path.basename(file_set['video'])))
            
            if file_set['csv1']:
                self.files_table.setItem(i, 2, QTableWidgetItem(os.path.basename(file_set['csv1'])))
            else:
                self.files_table.setItem(i, 2, QTableWidgetItem("None"))
                
            if file_set['csv2']:
                self.files_table.setItem(i, 3, QTableWidgetItem(os.path.basename(file_set['csv2'])))
            else:
                self.files_table.setItem(i, 3, QTableWidgetItem("None"))
        
        # Enable/disable the start button based on whether files were found
        self.start_batch_btn.setEnabled(len(file_sets) > 0)
    
    def table_double_clicked(self, index):
        """Handle double-click on file table to start processing that file."""
        if self.files_table.rowCount() > 0:
            # Get the row that was clicked
            row = index.row()
            # Switch to manual annotation tab
            self.tab_widget.setCurrentIndex(0)
            # Emit signal to load the selected file directly
            self.first_file_signal.emit()
    
    def start_batch_processing(self):
        """Start batch processing with the first file."""
        if self.files_table.rowCount() > 0:
            # Switch to manual annotation tab
            self.tab_widget.setCurrentIndex(0)
            # Emit signal to load the first file
            self.first_file_signal.emit()
    
    def load_next_file(self):
        """Load the next file in the batch."""
        self.next_file_signal.emit()
    
    def load_previous_file(self):
        """Load the previous file in the batch."""
        self.previous_file_signal.emit()
    
    def show_batch_complete_message(self):
        """Show a message when batch processing is complete."""
        QMessageBox.information(
            self,
            "Batch Processing Complete",
            "All files in the batch have been processed!"
        )
    
    def keyPressEvent(self, event):
        """
        Handle keyboard press events.
        """
        # Only process keyboard shortcuts in the manual annotation tab
        if self.tab_widget.currentIndex() == 0:
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
        # Only process keyboard shortcuts in the manual annotation tab
        if self.tab_widget.currentIndex() == 0:
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

    def setup_action_display(self):
        """
        Setup method to be called during player integration.
        This is an empty placeholder to be compatible with player_integration.py.
        """
        pass
