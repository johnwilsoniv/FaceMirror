# ui_manager.py
# --- START OF FILE ui_manager.py ---

from PyQt5.QtCore import QObject, pyqtSlot, QUrl
from PyQt5.QtWidgets import QMessageBox, QAction # Added QAction
from PyQt5.QtMultimedia import QMediaContent
import os
import config # Import config

class UIManager(QObject):
    """Manages UI updates and interactions for the MainWindow."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        if not self.window:
            raise ValueError("UIManager requires a valid MainWindow instance.")

    # (update_file_display, set_batch_navigation - unchanged)
    @pyqtSlot(str, str, str)
    def update_file_display(self, video_path, csv1_path, csv2_path):
        if not self.window: return; self.window.current_video_path = video_path; self.window.current_csv1_path = csv1_path; self.window.current_csv2_path = csv2_path
        self.window.video_path_label.setText(f"Video: {os.path.basename(video_path) if video_path else 'N/A'}"); self.window.csv_path_label.setText(f"CSV 1: {os.path.basename(csv1_path) if csv1_path else 'N/A'}"); self.window.second_csv_path_label.setText(f"CSV 2: {os.path.basename(csv2_path) if csv2_path else 'N/A'}")
    @pyqtSlot(bool, int, int)
    def set_batch_navigation(self, enabled, current_index, total_files):
        if not self.window: return;
        if total_files <= 0: self.window.batch_status_label.setText("No Files"); self.window.prev_file_btn.setEnabled(False); self.window.next_file_btn.setEnabled(False); self.window.save_btn.setText("Generate Output"); return
        self.window.batch_status_label.setText(f"File {current_index + 1} of {total_files}"); self.window.prev_file_btn.setEnabled(enabled and current_index > 0); self.window.next_file_btn.setEnabled(enabled and current_index < total_files - 1)
        if enabled: self.window.save_btn.setText("Save & Complete" if current_index >= total_files - 1 else "Save & Continue")
        else: self.window.save_btn.setText("Generate Output Files")

    @pyqtSlot(object)
    def update_action_display(self, ac_or_status):
        # (Unchanged from previous fix)
        if not self.window or not self.window.shared_action_display_label: return
        bg = config.UI_COLORS['section_bg']; tc = config.UI_COLORS['text_normal']; fw = "normal"; dt = "Status: Unknown"
        if isinstance(ac_or_status, str):
            if ac_or_status.startswith("Confirm:"): dt = ac_or_status; bg = "#fff3cd"; tc = "#856404"; fw = "bold"
            elif ac_or_status.startswith("Possible:"): dt = ac_or_status; bg = "#cfe2ff"; tc = "#0a367e"; fw = "bold"
            elif ac_or_status.startswith("Status:"):
                 dt = ac_or_status
                 if "error" in ac_or_status.lower(): bg = config.UI_COLORS.get('section_border', '#cccccc'); tc = config.DISCARD_BUTTON_STYLE.split("color: ")[1].split(";")[0]
            elif ac_or_status in config.ACTION_MAPPINGS:
                 action_code = ac_or_status
                 action_text = config.ACTION_MAPPINGS.get(action_code, f"Unknown ({action_code})")
                 if action_code == "TBC" or action_code == "NM":
                     dt = "Status: ??"
                     bg = config.UI_COLORS.get('timeline_tbc_color', '#eeeeee') if action_code == "TBC" else config.UI_COLORS.get('timeline_nm_color', '#ddeeff')
                     tc = config.UI_COLORS.get('timeline_tbc_nm_text', '#6c757d')
                     fw = "italic"
                 elif action_code == "BL":
                     dt = "Status: Baseline"
                     bg = config.UI_COLORS['neutral_status_bg']
                     tc = config.UI_COLORS['text_normal']
                     fw = "normal"
                 else: # Normal action codes
                     dt = f"{action_code}: {action_text}"
                     bg = config.UI_COLORS['highlight']
                     tc = "white"
                     fw = "bold"
            else: dt = f"Status: {ac_or_status}"
        elif ac_or_status is None: dt = "Status: Uncoded"; bg = config.UI_COLORS['neutral_status_bg']; tc = config.UI_COLORS['text_normal']; fw = "normal"
        else: dt = f"Status: Invalid Data ({type(ac_or_status).__name__})"; bg = config.DISCARD_BUTTON_STYLE.split("background-color: ")[1].split(";")[0]; tc = "white"
        stylesheet = f"QLabel{{background-color:{bg}; color:{tc}; border-radius:3px; padding:6px; min-height:30px; font-weight:{fw};}}"
        current_text = self.window.shared_action_display_label.text(); current_style = self.window.shared_action_display_label.styleSheet()
        if current_text != dt or current_style != stylesheet:
            self.window.shared_action_display_label.setText(dt); self.window.shared_action_display_label.setStyleSheet(stylesheet)


    # (update_frame_info, update_progress, show_progress_bar, enable_play_pause_button, set_play_button_state unchanged)
    @pyqtSlot(int, int)
    def update_frame_info(self, current_frame, total_frames):
        if not self.window or not self.window.frame_label or not self.window.frame_slider: return
        try: valid_total_frames = int(total_frames) if total_frames is not None else 0; valid_total_frames = max(0, valid_total_frames)
        except (TypeError, ValueError): valid_total_frames = 0; print(f"UIManager WARN: Invalid total_frames type received ({type(total_frames)}), defaulting to 0.")
        max_frame_index = max(0, valid_total_frames - 1); self.window.frame_label.setText(f"Frame: {current_frame}/{max_frame_index}")
        current_slider_max = self.window.frame_slider.maximum()
        if current_slider_max != max_frame_index: self.window.frame_slider.setMaximum(max_frame_index)
        if not self.window.frame_slider.isSliderDown():
            clamped_current_frame = max(0, min(current_frame, max_frame_index)); current_slider_value = self.window.frame_slider.value()
            if current_slider_value != clamped_current_frame: self.window.frame_slider.blockSignals(True); self.window.frame_slider.setValue(clamped_current_frame); self.window.frame_slider.blockSignals(False)
    @pyqtSlot(int)
    def update_progress(self, value):
        if self.window and self.window.progress_bar: self.window.progress_bar.setValue(value)
    @pyqtSlot(bool)
    def show_progress_bar(self, show):
        if self.window and self.window.progress_bar: self.window.progress_bar.setVisible(show);
        if show: self.window.progress_bar.setValue(0)
    @pyqtSlot(bool)
    def enable_play_pause_button(self, enable):
        if self.window and self.window.play_pause_btn: self.window.play_pause_btn.setEnabled(enable)
    @pyqtSlot(bool)
    def set_play_button_state(self, is_playing):
        if self.window and self.window.play_pause_btn: self.window.play_pause_btn.setText("Pause" if is_playing else "Play")

    # --- MODIFIED enable_action_buttons ---
    @pyqtSlot(bool, object, str)
    def enable_action_buttons(self, enable, current_action=None, context='idle'):
        """Enables/disables main action buttons, highlights if current."""
        if not self.window or not self.window.main_action_buttons: return

        is_prompt_active = context in ['confirm', 'near_miss']

        for code, button in self.window.main_action_buttons.items():
            # All buttons are enabled if a prompt is active or if general enable is true
            button_enabled = enable or is_prompt_active

            # Highlighting depends on context
            is_current = False
            if is_prompt_active:
                # No highlighting during prompts, all are potential choices
                is_current = False
            elif current_action is not None and code == current_action:
                # Highlight if it matches the selected range's action (non-prompt context)
                is_current = True

            style = config.PRIMARY_BUTTON_STYLE if is_current else config.STANDARD_BUTTON_STYLE
            button.setEnabled(button_enabled)
            button.setStyleSheet(style if button_enabled else config.DISABLED_BUTTON_STYLE)

    @pyqtSlot(bool)
    def enable_clear_button(self, enable):
        if self.window and self.window.clear_all_button: self.window.clear_all_button.setEnabled(enable)

    @pyqtSlot(bool)
    def enable_undo_button(self, enable):
        if self.window and self.window.undo_button:
            self.window.undo_button.setEnabled(enable)
            style = config.STANDARD_BUTTON_STYLE if enable else config.DIMMED_UTILITY_BUTTON_STYLE
            self.window.undo_button.setStyleSheet(style)

    @pyqtSlot(bool)
    def enable_redo_button(self, enable):
        if self.window and self.window.redo_button:
            self.window.redo_button.setEnabled(enable)
            style = config.STANDARD_BUTTON_STYLE if enable else config.DIMMED_UTILITY_BUTTON_STYLE
            self.window.redo_button.setStyleSheet(style)

    @pyqtSlot(bool)
    def enable_delete_button(self, enable):
        if self.window and self.window.delete_button:
            self.window.delete_button.setEnabled(enable)

    # --- NEW METHODS for Discard Buttons ---
    @pyqtSlot(bool)
    def show_discard_confirmation_button(self, show):
        if self.window and self.window.discard_confirmation_button:
            self.window.discard_confirmation_button.setVisible(show)

    @pyqtSlot(bool)
    def show_discard_near_miss_button(self, show):
        if self.window and self.window.discard_near_miss_button:
            self.window.discard_near_miss_button.setVisible(show)
    # --- END NEW METHODS ---

    @pyqtSlot(str)
    def play_audio_snippet(self, audio_path): # (Unchanged)
        if not self.window or not self.window.snippet_player: return
        if not os.path.exists(audio_path): print(f"UIManager WARN: Snippet path not found: {audio_path}"); return
        try:
            media_content = QMediaContent(QUrl.fromLocalFile(os.path.abspath(audio_path)))
            if media_content.isNull(): print(f"UIManager ERROR: Could not create valid QMediaContent for {audio_path}"); return
            self.window.snippet_player.setMedia(media_content); self.window.snippet_player.setVolume(80); self.window.snippet_player.play()
        except Exception as e: print(f"UIManager ERROR playing snippet: {e}")

    def show_batch_complete_message(self): # (Unchanged)
        if self.window: QMessageBox.information(self.window, "Batch Complete", "All files processed!")

    def show_message_box(self, level, title, message): # (Unchanged)
        if not self.window: return;
        if level == "info": QMessageBox.information(self.window, title, message)
        elif level == "warning": QMessageBox.warning(self.window, title, message)
        elif level == "critical": QMessageBox.critical(self.window, title, message)
        else: QMessageBox.information(self.window, title, message)

# --- END OF ui_manager.py ---