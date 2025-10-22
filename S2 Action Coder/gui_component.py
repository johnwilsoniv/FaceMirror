
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QSlider, QGroupBox, # Removed QFileDialog
                           QProgressBar, QMessageBox, QSplitter, QSizePolicy, QApplication,
                           QListWidget, QListWidgetItem, QRadioButton,
                           QButtonGroup, QCheckBox, QTableWidget, QTableWidgetItem,
                           QHeaderView, QAbstractItemView, QGridLayout, # Removed QStackedWidget
                           QSpacerItem, QAction) # Added QAction for shortcuts
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot, QEvent, QUrl
from PyQt5.QtGui import QPixmap, QFont, QKeySequence # Keep QFont, Add QKeySequence
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import config
import os

from timeline_widget import TimelineWidget

class MainWindow(QMainWindow):
    # --- REMOVED OLD PROMPT SIGNALS ---
    frame_changed_signal = pyqtSignal(int); play_pause_signal = pyqtSignal(bool); save_signal = pyqtSignal(); progress_update_signal = pyqtSignal(int)
    next_file_signal = pyqtSignal(); previous_file_signal = pyqtSignal()
    clear_manual_annotations_signal = pyqtSignal()
    main_action_button_clicked = pyqtSignal(str)
    delete_selected_range_signal = pyqtSignal()


    def __init__(self):
        # (Initialization remains the same)
        super().__init__(); self.setWindowTitle("Facial Action Coder"); self.setMinimumSize(1200, 800) # Adjusted min height slightly
        self.current_video_path=None; self.current_csv1_path=None; self.current_csv2_path=None
        self.output_csv_path=None; self.output_second_csv_path=None; self.output_video_path=None
        self.video_display_widget=None; self.play_pause_btn=None; self.frame_slider=None
        self.frame_label=None; self.video_path_label=None; self.csv_path_label=None; self.second_csv_path_label=None
        self.prev_file_btn=None; self.next_file_btn=None; self.batch_status_label=None
        self.save_btn=None; self.progress_bar=None; self.shared_action_display_label=None
        # --- REMOVED INTERACTION STACK & PANELS ---
        self.timeline_widget = None; self.snippet_player = QMediaPlayer(); self._default_label_palette = None
        self.action_buttons_panel = None; self.main_action_buttons = {}
        self.utility_buttons_panel = None; self.clear_all_button = None;
        self.undo_button = None; self.redo_button = None
        self.delete_button = None
        # Discard Confirmation and Discard Near Miss buttons removed - Delete Range button serves same purpose
        self.setup_ui()
        self._setup_menu_actions() # Call setup for menu actions
        self.progress_update_signal.connect(self.update_progress)

    def setup_ui(self):
         # (Layout setup remains the same as previous corrected version)
         central_widget = QWidget(); main_layout = QVBoxLayout(central_widget)
         main_layout.setContentsMargins(config.STANDARD_MARGIN, config.STANDARD_MARGIN, config.STANDARD_MARGIN, config.STANDARD_MARGIN); main_layout.setSpacing(config.STANDARD_SPACING * 2)
         top_section_widget = QWidget(); top_section_layout = QHBoxLayout(top_section_widget); top_section_layout.setContentsMargins(0,0,0,0)
         video_area_widget = QWidget(); video_area_layout = QVBoxLayout(video_area_widget); video_area_layout.setContentsMargins(0,0,0,0); video_area_layout.setSpacing(0)
         self.video_display_widget = QLabel("Video Player Area"); self.video_display_widget.setObjectName("videoPlaceholder")
         self.video_display_widget.setAlignment(Qt.AlignCenter); self.video_display_widget.setMinimumSize(480, 360); self.video_display_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.video_display_widget.setStyleSheet("background-color: black; color: white;")
         video_area_layout.addWidget(self.video_display_widget, 1); top_section_layout.addWidget(video_area_widget, 2)
         right_panel_widget = QWidget(); right_panel_widget.setMaximumWidth(400); right_panel_layout = QVBoxLayout(right_panel_widget); right_panel_layout.setContentsMargins(config.STANDARD_MARGIN,0,0,0); right_panel_layout.setSpacing(config.STANDARD_SPACING * 2)
         file_info_group = QGroupBox("Current Files"); file_info_group.setStyleSheet(config.GROUP_BOX_STYLE); file_info_layout = QVBoxLayout(file_info_group); file_info_layout.setSpacing(config.STANDARD_SPACING)
         self.video_path_label = QLabel("Video: N/A"); self.video_path_label.setWordWrap(True); self.csv_path_label = QLabel("CSV 1: N/A"); self.csv_path_label.setWordWrap(True); self.second_csv_path_label = QLabel("CSV 2: N/A"); self.second_csv_path_label.setWordWrap(True); file_info_layout.addWidget(self.video_path_label); file_info_layout.addWidget(self.csv_path_label); file_info_layout.addWidget(self.second_csv_path_label); right_panel_layout.addWidget(file_info_group)
         batch_nav_group = QGroupBox("Batch Navigation"); batch_nav_group.setStyleSheet(config.GROUP_BOX_STYLE); batch_nav_layout_hbox = QHBoxLayout(batch_nav_group); batch_nav_layout_hbox.setSpacing(config.STANDARD_SPACING)
         self.prev_file_btn = QPushButton("← Previous"); self.prev_file_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE); self.prev_file_btn.clicked.connect(self.load_previous_file); self.prev_file_btn.setEnabled(False); batch_nav_layout_hbox.addWidget(self.prev_file_btn)
         self.batch_status_label = QLabel("File 1 of 1"); self.batch_status_label.setAlignment(Qt.AlignCenter); batch_nav_layout_hbox.addWidget(self.batch_status_label, 1)
         self.next_file_btn = QPushButton("Next →"); self.next_file_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE); self.next_file_btn.clicked.connect(self.load_next_file); self.next_file_btn.setEnabled(False); batch_nav_layout_hbox.addWidget(self.next_file_btn)
         right_panel_layout.addWidget(batch_nav_group)
         action_display_group = QGroupBox("Current Action / Status"); action_display_group.setStyleSheet(config.GROUP_BOX_STYLE); action_display_layout = QVBoxLayout(action_display_group)
         self.shared_action_display_label = QLabel("Status: Initializing..."); self.shared_action_display_label.setAlignment(Qt.AlignCenter); self.shared_action_display_label.setStyleSheet(f"QLabel{{background-color:{config.UI_COLORS['section_bg']}; color:{config.UI_COLORS['text_normal']}; border-radius:3px; padding:6px; min-height:30px; font-weight:normal;}}")
         action_display_layout.addWidget(self.shared_action_display_label)
         right_panel_layout.addWidget(action_display_group)
         playback_controls_group = QGroupBox("Playback Controls"); playback_controls_group.setStyleSheet(config.GROUP_BOX_STYLE)
         controls_layout = QHBoxLayout(playback_controls_group); controls_layout.setContentsMargins(5, 8, 5, 8); controls_layout.setSpacing(config.STANDARD_SPACING)
         self.play_pause_btn = QPushButton("Play"); self.play_pause_btn.setFixedWidth(60); self.play_pause_btn.clicked.connect(self.toggle_play_pause); self.play_pause_btn.setStyleSheet(config.STANDARD_BUTTON_STYLE); controls_layout.addWidget(self.play_pause_btn)
         self.frame_slider = QSlider(Qt.Horizontal); self.frame_slider.setTickPosition(QSlider.TicksBelow); self.frame_slider.valueChanged.connect(self.slider_changed); self.frame_slider.setStyleSheet(config.SLIDER_STYLE); controls_layout.addWidget(self.frame_slider, 1)
         self.frame_label = QLabel("Frame: 0/0"); self.frame_label.setFixedWidth(100); controls_layout.addWidget(self.frame_label)
         right_panel_layout.addWidget(playback_controls_group); right_panel_layout.addStretch(1); top_section_layout.addWidget(right_panel_widget, 1)
         main_layout.addWidget(top_section_widget, 1)
         self.timeline_widget = TimelineWidget(); self.timeline_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
         main_layout.addWidget(self.timeline_widget, 1)
         button_section_layout = QHBoxLayout(); button_section_layout.setContentsMargins(0, config.STANDARD_MARGIN, 0, 0); button_section_layout.setSpacing(config.STANDARD_SPACING * 3)
         self._setup_action_buttons_panel(); button_section_layout.addWidget(self.action_buttons_panel, 1)
         self._setup_utility_buttons_panel(); button_section_layout.addWidget(self.utility_buttons_panel, 0)
         main_layout.addLayout(button_section_layout, 0)
         main_layout.addStretch(1)
         bottom_bar_layout = QHBoxLayout(); bottom_bar_layout.setContentsMargins(0, config.STANDARD_MARGIN // 2, 0, 0)
         self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0); self.progress_bar.setVisible(False); self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
         bottom_bar_layout.addWidget(self.progress_bar, 1)
         self.save_btn = QPushButton("Generate Output Files"); self.save_btn.setMinimumHeight(35); self.save_btn.setStyleSheet(config.PRIMARY_BUTTON_STYLE); self.save_btn.clicked.connect(self.save_outputs)
         bottom_bar_layout.addWidget(self.save_btn)
         main_layout.addLayout(bottom_bar_layout, 0)
         self.setCentralWidget(central_widget)

    # --- MODIFIED _setup_action_buttons_panel ---
    def _setup_action_buttons_panel(self):
        self.action_buttons_panel = QWidget()
        layout = QGridLayout(self.action_buttons_panel)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(config.STANDARD_SPACING)
        self.main_action_buttons = {}

        # Define the desired order and grouping of action codes
        action_groups = [
            # Group 1: Eyes / Brows
            "RE", "FR", "ES", "ET", "BK",
            # Group 2: Smiles
            "SS", "BS",
             # Group 3: Mouth / Lips / Teeth / Cheeks
            "PL", "LT", "BC", "SO", "SE",
            # Group 4: Nose
            "WN",
            # Group 5: Baseline (If needed as a button, though usually not)
            # "BL"
        ]
        # Ensure all relevant codes from config are included (handles potential future additions)
        all_codes_in_config = [code for code in config.ACTION_MAPPINGS if code not in ["STOP", "SO_SE", "TBC", "NM"]]
        ordered_codes = [code for code in action_groups if code in all_codes_in_config]
        # Add any codes from config not explicitly listed in groups (maintaining some order)
        for code in sorted(all_codes_in_config):
            if code not in ordered_codes:
                ordered_codes.append(code)

        row, col, max_cols = 0, 0, 5 # Adjust max_cols as needed for layout

        for action_code in ordered_codes:
            action_text = config.ACTION_MAPPINGS.get(action_code, f"Unknown ({action_code})")
            if not action_text: # Skip if mapping somehow missing
                continue

            # Format button text: "Action Text [CODE]"
            button_text = f"{action_text} [{action_code}]"
            btn = QPushButton(button_text)
            btn.setStyleSheet(config.DISABLED_BUTTON_STYLE)
            btn.setEnabled(False)
            btn.setToolTip(f"Assign '{action_text}' [{action_code}] to selected range")

            # Use lambda to capture the correct action_code for the signal
            btn.clicked.connect(lambda checked=False, code=action_code: self.main_action_button_clicked.emit(code))

            self.main_action_buttons[action_code] = btn
            layout.addWidget(btn, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    # --- END MODIFIED ---

    # (Rest of the methods remain the same: _setup_utility_buttons_panel, _setup_menu_actions, UI updates, slots etc.)
    def _setup_utility_buttons_panel(self):
        self.utility_buttons_panel = QWidget()
        layout = QVBoxLayout(self.utility_buttons_panel)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(config.STANDARD_SPACING)
        layout.setAlignment(Qt.AlignTop) # Keep alignment

        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.setToolTip("Clear all annotations for this file")
        self.clear_all_button.setStyleSheet(config.DISCARD_BUTTON_STYLE)
        self.clear_all_button.clicked.connect(self.clear_manual_annotations_signal.emit) # Signal to controller
        self.clear_all_button.setEnabled(False)
        layout.addWidget(self.clear_all_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Undo last action (Ctrl+Z)") # Tooltip updated
        self.undo_button.setStyleSheet(config.STANDARD_BUTTON_STYLE)
        self.undo_button.setEnabled(False) # Initially disabled
        layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.setToolTip("Redo last undone action (Ctrl+Shift+Z)") # Tooltip updated
        self.redo_button.setStyleSheet(config.STANDARD_BUTTON_STYLE)
        self.redo_button.setEnabled(False) # Initially disabled
        layout.addWidget(self.redo_button)

        self.delete_button = QPushButton("Delete Range")
        self.delete_button.setToolTip("Delete the currently selected range (Delete/Backspace)")
        self.delete_button.setStyleSheet(config.DELETE_BUTTON_STYLE)
        self.delete_button.clicked.connect(self.delete_selected_range_signal.emit) # Signal to controller
        self.delete_button.setEnabled(False) # Initially disabled
        layout.addWidget(self.delete_button)

        # Discard Confirmation and Discard Near Miss buttons removed - Delete Range button serves same purpose

        layout.addStretch(1) # Push buttons up

    def _setup_menu_actions(self): # (Unchanged)
        self.undo_action = QAction('&Undo', self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.setEnabled(False) # Initial state
        if self.undo_button:
            self.undo_action.triggered.connect(self.undo_button.click)
        self.redo_action = QAction('&Redo', self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.setEnabled(False) # Initial state
        if self.redo_button:
            self.redo_action.triggered.connect(self.redo_button.click)
        menubar = self.menuBar()
        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        if self.undo_button:
            self.undo_button.setEnabled = self._wrap_set_enabled(self.undo_button.setEnabled, self.undo_action)
        if self.redo_button:
            self.redo_button.setEnabled = self._wrap_set_enabled(self.redo_button.setEnabled, self.redo_action)

    def _wrap_set_enabled(self, original_set_enabled_method, menu_action): # (Unchanged)
        def wrapper(enabled):
            original_set_enabled_method(enabled)
            if menu_action:
                menu_action.setEnabled(enabled)
        return wrapper

    @pyqtSlot(str,str,str)
    def update_file_display(self,vp,c1,c2): # (Unchanged)
        self.current_video_path=vp;self.current_csv1_path=c1;self.current_csv2_path=c2; self.video_path_label.setText(f"Video: {os.path.basename(vp) if vp else 'N/A'}"); self.csv_path_label.setText(f"CSV 1: {os.path.basename(c1) if c1 else 'N/A'}"); self.second_csv_path_label.setText(f"CSV 2: {os.path.basename(c2) if c2 else 'N/A'}")
    @pyqtSlot(bool,int,int)
    def set_batch_navigation(self,en,ci,tf): # (Unchanged)
         if tf<=0: self.batch_status_label.setText("No Files"); self.prev_file_btn.setEnabled(False); self.next_file_btn.setEnabled(False); self.save_btn.setText("Generate Output"); return
         self.batch_status_label.setText(f"File {ci+1} of {tf}"); self.prev_file_btn.setEnabled(en and ci>0); self.next_file_btn.setEnabled(en and ci<tf-1)
         if en: self.save_btn.setText("Save and Complete" if ci>=tf-1 else "Save and Continue")
         else: self.save_btn.setText("Generate Output Files")
    @pyqtSlot(bool)
    def enable_play_pause_button(self,en): # (Unchanged)
         if self.play_pause_btn: self.play_pause_btn.setEnabled(en)
    @pyqtSlot(int, QPixmap, str) # (Unchanged)
    def update_video_frame(self, frame_number, pixmap, action_code):
          if not self.video_display_widget: return
          if isinstance(self.video_display_widget, QLabel) and self.video_display_widget.objectName() == "videoPlaceholder":
              if pixmap and not pixmap.isNull():
                  pw = self.video_display_widget.width(); ph = self.video_display_widget.height()
                  if pw > 0 and ph > 0: self.video_display_widget.setPixmap(pixmap.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                  else: self.video_display_widget.setPixmap(pixmap)
              else: self.video_display_widget.setText(f"Frame: {frame_number}\n(No Image Data)")
          elif isinstance(self.video_display_widget, QWidget):
               pass # Player integration handles display
    @pyqtSlot(int,int)
    def update_frame_info(self, current_frame, total_frames): # (Unchanged)
          max_frame_index = max(0, total_frames - 1); self.frame_label.setText(f"Frame: {current_frame}/{max_frame_index}")
          if self.frame_slider.maximum() != max_frame_index: self.frame_slider.setMaximum(max_frame_index)
          if not self.frame_slider.isSliderDown():
             clamped_value = max(0, min(current_frame, max_frame_index))
             if self.frame_slider.value() != clamped_value:
                 self.frame_slider.blockSignals(True); self.frame_slider.setValue(clamped_value); self.frame_slider.blockSignals(False)
    @pyqtSlot(object)
    def update_action_display(self, ac): # (Unchanged - handled by UIManager now)
        pass # Handled by UIManager now
    def set_total_frames(self,tf): # (Unchanged)
        mfi=max(0,tf-1); self.frame_slider.setMaximum(mfi); cf=self.frame_slider.value(); self.frame_label.setText(f"Frame: {cf}/{mfi}")
    def toggle_play_pause(self):
        # Optimistic UI update for instant feedback
        should_play = (self.play_pause_btn.text() == "Play")
        # Immediately update button text and disable to prevent double-clicks
        self.play_pause_btn.setText("Pause" if should_play else "Play")
        self.play_pause_btn.setEnabled(False)
        # Emit signal to controller
        self.play_pause_signal.emit(should_play)

    @pyqtSlot(bool)
    def set_play_button_state(self, isp):
        # Update button text (in case state was reverted due to error)
        self.play_pause_btn.setText("Pause" if isp else "Play")
        # Re-enable button now that state change is confirmed
        self.play_pause_btn.setEnabled(True)
    def slider_changed(self,v): # (Unchanged)
        self.frame_changed_signal.emit(v)
    @pyqtSlot(int)
    def update_progress(self,v): # (Unchanged)
        self.progress_bar.setValue(v)
    def show_progress_bar(self,v=True): # (Unchanged)
        self.progress_bar.setVisible(v); self.progress_bar.setValue(0)
    def save_outputs(self): # (Unchanged)
        self.save_signal.emit()
    def load_next_file(self): # (Unchanged)
        self.next_file_signal.emit()
    def load_previous_file(self): # (Unchanged)
        self.previous_file_signal.emit()
    def show_batch_complete_message(self): # (Unchanged)
        QMessageBox.information(self,"Batch Complete","All files processed!")
    @pyqtSlot(str)
    def play_audio_snippet(self,ap): # (Unchanged)
           if not os.path.exists(ap): print(f"Warn: Snippet path not found: {ap}"); return
           try: self.snippet_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(ap)))); self.snippet_player.setVolume(80); self.snippet_player.play()
           except Exception as e: print(f"Error playing snippet: {e}")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for main window"""
        # Check modifiers
        modifiers = event.modifiers()

        if event.key() == Qt.Key_Space:
            # Spacebar triggers play/pause
            if self.play_pause_btn and self.play_pause_btn.isEnabled():
                self.toggle_play_pause()
                event.accept()
                return
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # Enter key triggers save
            if self.save_btn and self.save_btn.isEnabled():
                self.save_outputs()
                event.accept()
                return
        elif event.key() == Qt.Key_Z and modifiers == Qt.ControlModifier:
            # Ctrl+Z triggers undo
            if self.undo_button and self.undo_button.isEnabled():
                self.undo_button.click()
                event.accept()
                return
        elif event.key() == Qt.Key_Z and modifiers == (Qt.ControlModifier | Qt.ShiftModifier):
            # Ctrl+Shift+Z triggers redo
            if self.redo_button and self.redo_button.isEnabled():
                self.redo_button.click()
                event.accept()
                return
        # Pass other keys to parent
        super().keyPressEvent(event)
    def closeEvent(self,evt): # (Unchanged)
        print("Closing..."); self.snippet_player.stop(); evt.accept()

