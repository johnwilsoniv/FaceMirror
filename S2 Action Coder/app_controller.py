# --- START OF FILE app_controller.py ---

import sys
import os
import tempfile # For audio snippets
import subprocess # For ffmpeg
import gc
from PyQt5.QtCore import QObject, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel
from PyQt5.QtMultimedia import QMediaPlayer

# Import project modules
import config_paths
from qt_media_player import QTMediaPlayer
from gui_component import MainWindow
from action_tracker import ActionTracker
from csv_handler import CSVHandler
from player_integration import integrate_qt_player
from batch_processor import BatchProcessor
from timeline_processor import TimelineProcessor
import config
from ui_manager import UIManager
from playback_manager import PlaybackManager
from processing_manager import ProcessingManager
from history_manager import HistoryManager
import native_dialogs
import copy
import traceback
import time
import sys

class ApplicationController(QObject):
    def __init__(self, app_instance, whisper_model=None):
        # Display version information
        print(f"\n{'='*60}")
        print(f"{config_paths.APP_NAME} v{config_paths.VERSION}")
        print(f"{'='*60}\n")
        super().__init__()
        self.app = app_instance
        self.preloaded_whisper_model = whisper_model  # Store pre-loaded model
        self.window = None
        self.video_player = QTMediaPlayer()
        self.action_tracker = ActionTracker()
        self.csv_handler = CSVHandler()
        self.batch_processor = BatchProcessor()
        self.timeline_processor = TimelineProcessor(config)
        self.ffmpeg_path = config_paths.get_ffmpeg_path()
        self.ui_manager = None
        self.playback_manager = None
        self.processing_manager = None
        self.history_manager = HistoryManager(self)
        self.current_file_set = None
        self.whisper_processed = False
        self.final_timeline_events = []
        self.generated_action_ranges = []
        # --- Prompt State ---
        self.waiting_for_confirmation = False
        self.current_confirmation_info = None
        self.waiting_for_near_miss_assignment = False
        self.current_near_miss_info = None
        self.active_prompt_audio_snippet_path = None # Path to temporary snippet file
        self.pending_action_for_creation = None
        # --- End Prompt State ---
        self.next_timeline_event_index = 0
        self.selected_timeline_range_data = None
        self.current_selected_index = -1
        # Batch processing tracking
        self.batch_start_time = None
        self.failed_files = []  # List of failed file base_ids
        self.output_directory = None  # Track output directory for completion summary
        if not self._run_startup_sequence():
            if self.window: self.window.close()
            sys.exit(0)
        self._initialize_managers()
        integrate_qt_player(self.window, self.video_player)
        self._connect_signals()
        self.load_first_file()
        self.window.show()
        self.window.activateWindow()

    # (_run_startup_sequence, _initialize_managers, _connect_signals - unchanged)
    def _run_startup_sequence(self): # (Unchanged)
        file_dialog=QFileDialog(None,"Select Video Files or a Directory"); file_dialog.setFileMode(QFileDialog.ExistingFiles); file_dialog.setOption(QFileDialog.DontUseNativeDialog,False); file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv)")
        # Set default directory to S1 output folder
        default_dir = config_paths.get_s1_processed_dir()
        if default_dir.exists():
            file_dialog.setDirectory(str(default_dir))
        processed_suffixes = ["_annotated", "_coded"];
        if file_dialog.exec_():
            selected_files=file_dialog.selectedFiles(); selected_dir=file_dialog.directory().absolutePath(); valid_files = []; rejected_files = []
            for f in selected_files:
                basename = os.path.basename(f)
                if any(suffix.lower() in basename.lower() for suffix in processed_suffixes): rejected_files.append(basename)
                else: valid_files.append(f)
            if rejected_files: native_dialogs.show_warning("Invalid Input Files", "Excluded potentially processed files:\n" + "\n".join(rejected_files))
            if not valid_files and not selected_dir: native_dialogs.show_warning("No Valid Files", "No valid video files selected or directory chosen."); return False
            selected_files = valid_files; file_sets = []
            if not selected_files:
                 if selected_dir: file_sets=self.batch_processor.find_matching_files(selected_dir)
                 if not file_sets: native_dialogs.show_warning("No Valid Files", f"No valid Video/CSV sets found in {selected_dir or 'selection'}."); return False
            else:
                # Determine search directory for CSVs
                video_dir = os.path.dirname(selected_files[0])
                search_dir = video_dir

                # Check if videos are in "Face Mirror 1.0 Output" - if so, look for CSVs in sibling "Combined Data"
                if os.path.basename(video_dir) == "Face Mirror 1.0 Output":
                    parent_dir = os.path.dirname(video_dir)
                    combined_data_dir = os.path.join(parent_dir, "Combined Data")
                    if os.path.isdir(combined_data_dir):
                        search_dir = combined_data_dir
                        print(f"S1 integration: Videos in 'Face Mirror 1.0 Output', searching for CSVs in 'Combined Data'")

                file_sets=self.batch_processor.find_matching_files_for_videos(selected_files, search_dir)
                if not file_sets: native_dialogs.show_warning("No Matches", f"Could not find matching CSVs for the selected videos.\nSearched in: {search_dir}"); return False

            # Check for existing outputs before proceeding
            # Calculate output directory (same logic as in _initiate_save)
            # Track whether user already confirmed via output detection dialog
            user_already_confirmed = False
            if file_sets:
                first_video = file_sets[0].get('video')
                if first_video:
                    current_video_dir = os.path.dirname(first_video)
                    batch_parent_dir = os.path.dirname(current_video_dir)
                    if not batch_parent_dir or batch_parent_dir == current_video_dir:
                        batch_parent_dir = current_video_dir
                    # Always use standard output location from config_paths
                    output_dir = str(config_paths.get_output_base_dir())

                    # Check for existing outputs
                    output_check = self.batch_processor.check_existing_outputs(file_sets, output_dir)

                    if output_check['processed_count'] > 0:
                        # Some files are already processed, show 3-option dialog
                        total = len(file_sets)
                        processed_count = output_check['processed_count']
                        unprocessed_count = output_check['unprocessed_count']

                        dialog_msg = f"Found {total} file sets:\n• {processed_count} already processed\n• {unprocessed_count} not yet processed\n\nWhat would you like to do?"

                        user_choice = native_dialogs.ask_three_choice(
                            "Existing Outputs Detected",
                            dialog_msg,
                            "Process All",
                            "Skip Processed",
                            "Cancel",
                            default_button=2  # Default to "Skip Processed"
                        )

                        if user_choice == 1:
                            # Process All - keep all file_sets
                            print(f"User chose: Process All ({total} files)")
                            user_already_confirmed = True
                        elif user_choice == 2:
                            # Skip Processed - keep only unprocessed
                            file_sets = output_check['unprocessed']
                            print(f"User chose: Skip Processed ({len(file_sets)} files)")
                            user_already_confirmed = True
                            if not file_sets:
                                native_dialogs.show_info("Nothing to Process", "All files have already been processed.")
                                return False
                        else:
                            # Cancel or None
                            print("User cancelled at existing outputs dialog")
                            return False

            self.batch_processor.set_file_sets(file_sets)
        else: return False

        # Only show confirmation dialog if user hasn't already confirmed via output detection dialog
        if not user_already_confirmed:
            # Build file list for confirmation (limit to first 10 for large batches)
            total_files = len(self.batch_processor.file_sets)
            if total_files <= 10:
                # Show all files
                file_list_str = "\n".join([f"- {os.path.basename(fs['video'])}" for fs in self.batch_processor.file_sets])
            else:
                # Show first 10 + count of remaining
                first_10 = "\n".join([f"- {os.path.basename(fs['video'])}" for fs in self.batch_processor.file_sets[:10]])
                remaining = total_files - 10
                file_list_str = f"{first_10}\n...and {remaining} more"

            confirm_msg = f"Found {total_files} valid sets:\n{file_list_str}\n\nProceed?"
            print(f"DEBUG: About to show confirmation dialog for {total_files} files")
            try:
                user_response = native_dialogs.ask_yes_no("Confirm Files", confirm_msg, default_yes=True)
                print(f"DEBUG: User response to confirmation: {user_response}")
                if not user_response:
                    print("DEBUG: User declined confirmation, returning False")
                    return False
            except Exception as e:
                print(f"ERROR: Native dialog failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"DEBUG: Skipping redundant confirmation dialog (user already confirmed via output detection)")

        self.window=MainWindow(); return True
    def _initialize_managers(self): # (Unchanged)
        if not self.window: raise RuntimeError("Cannot initialize managers without a MainWindow.")
        print("Controller: Initializing UIManager...")
        try: self.ui_manager = UIManager(self.window, self); print(f"Controller: UIManager initialized: {self.ui_manager is not None}")
        except Exception as e: print(f"ERROR Initializing UIManager: {e}"); self.ui_manager = None
        print("Controller: Initializing PlaybackManager...")
        try: self.playback_manager = PlaybackManager(self.video_player, self); print(f"Controller: PlaybackManager initialized: {self.playback_manager is not None}")
        except Exception as e: print(f"ERROR Initializing PlaybackManager: {e}"); self.playback_manager = None
        print("Controller: Initializing ProcessingManager...")
        try: self.processing_manager = ProcessingManager(self.timeline_processor, self.ffmpeg_path, self, whisper_model=self.preloaded_whisper_model); print(f"Controller: ProcessingManager initialized: {self.processing_manager is not None}")
        except Exception as e: print(f"ERROR Initializing ProcessingManager: {e}"); self.processing_manager = None
        if not all([self.ui_manager, self.playback_manager, self.processing_manager]): print("ERROR: One or more managers failed to initialize properly!")
        else: print("Controller: All Managers initialized successfully.")
    def _connect_signals(self): # (Unchanged)
        if not all([self.ui_manager, self.playback_manager, self.processing_manager]): raise RuntimeError("Managers must be initialized before connecting signals.")
        print("Controller: Connecting signals...")
        self.window.play_pause_signal.connect(self._toggle_playback)
        self.window.frame_changed_signal.connect(self.playback_manager.seek)
        self.window.save_signal.connect(self._initiate_save)
        self.window.next_file_signal.connect(self.load_next_file)
        self.window.previous_file_signal.connect(self.load_previous_file)
        self.window.clear_manual_annotations_signal.connect(self._handle_clear_manual_annotations)
        self.window.main_action_button_clicked.connect(self._handle_main_action_button_clicked)
        self.window.delete_selected_range_signal.connect(self._handle_delete_selected_range_button)
        if self.window and self.window.discard_confirmation_button:
            self.window.discard_confirmation_button.clicked.connect(self._handle_discard_confirmation_prompt)
            print("Controller: Connected discard_confirmation_button")
        if self.window and self.window.discard_near_miss_button:
            self.window.discard_near_miss_button.clicked.connect(self._handle_discard_near_miss_prompt)
            print("Controller: Connected discard_near_miss_button")
        self.playback_manager.playback_state_changed.connect(self.ui_manager.set_play_button_state)
        self.playback_manager.frame_update_needed.connect(self._handle_frame_update)
        if self.window.timeline_widget:
            self.playback_manager.frame_update_needed.connect(self.window.timeline_widget.set_current_frame)
        self.playback_manager.video_ended.connect(self._handle_video_finished)
        self.playback_manager.playback_error.connect(lambda msg: self.ui_manager.show_message_box("critical", "Playback Error", msg))
        self.processing_manager.processing_status_update.connect(self._handle_processing_status_update)
        self.processing_manager.processing_progress_update.connect(self.ui_manager.update_progress)
        self.processing_manager.whisper_results_ready.connect(self._handle_whisper_results)
        self.processing_manager.save_complete.connect(self._handle_save_complete)
        self.processing_manager.processing_error.connect(self._handle_processing_error)
        self.action_tracker.action_ranges_changed.connect(self._update_timeline_widget_ranges)
        if self.window.timeline_widget:
            self.window.timeline_widget.ranges_edited.connect(self._handle_timeline_ranges_edited)
            self.window.timeline_widget.delete_range_requested.connect(self._handle_timeline_delete_request)
            self.window.timeline_widget.seek_requested.connect(self.playback_manager.seek)
            self.window.timeline_widget.range_selected.connect(self._handle_timeline_range_selected)
        self.history_manager.state_restored.connect(self._apply_restored_state)
        self.history_manager.history_changed.connect(self._update_undo_redo_buttons)
        if self.window and self.window.undo_button:
            self.window.undo_button.clicked.connect(self._do_undo)
            print("Controller: Connected undo_button")
        else:
            print("Controller WARN: Undo button not found in window.")
        if self.window and self.window.redo_button:
            self.window.redo_button.clicked.connect(self._do_redo)
            print("Controller: Connected redo_button")
        else:
            print("Controller WARN: Redo button not found in window.")
        print("Controller: Signals connected.")
        self._update_undo_redo_buttons()

    # (load_batch_file_set, load_first_file, load_next_file, load_previous_file, _toggle_playback, _handle_frame_update, _get_action_for_display - unchanged from previous updates)
    def load_batch_file_set(self, file_set): # (Unchanged)
        if not file_set or not self.window: return False;
        print(f"Controller: Loading file set: {file_set.get('base_id', 'N/A')}");
        self.current_file_set = file_set;
        self.processing_manager.stop_whisper_processing(); self.processing_manager.cleanup_previous_whisper_files()
        if self.playback_manager.is_playing: self.playback_manager.pause()
        self.action_tracker.clear_actions(); self.history_manager.clear();
        self.whisper_processed = False; self.final_timeline_events = []; self.generated_action_ranges = [];
        self.waiting_for_confirmation = False; self.current_confirmation_info = None
        self.waiting_for_near_miss_assignment = False; self.current_near_miss_info = None
        self._cleanup_active_snippet()
        self.pending_action_for_creation = None
        if self.ui_manager: self.ui_manager.clear_pending_action_button()
        self.next_timeline_event_index = 0;
        self.selected_timeline_range_data = None; self.current_selected_index = -1;
        self.ui_manager.update_action_display("Status: Initializing..."); self.ui_manager.update_file_display( file_set.get('video',''), file_set.get('csv1',''), file_set.get('csv2','')); self.ui_manager.set_batch_navigation( True, self.batch_processor.get_current_index(), self.batch_processor.get_total_files() ); self.ui_manager.set_play_button_state(False); self.ui_manager.enable_play_pause_button(False)
        if self.ui_manager:
            self.ui_manager.show_discard_confirmation_button(False)
            self.ui_manager.show_discard_near_miss_button(False)
        if self.window.timeline_widget: self.window.timeline_widget.set_editing_enabled(False); print("Controller: Timeline editing DISABLED."); self.window.timeline_widget._selected_range_index = -1; self.window.timeline_widget.update()
        self.ui_manager.enable_action_buttons(False); self.ui_manager.enable_clear_button(False);
        self.ui_manager.enable_delete_button(False)
        video_path=file_set.get('video'); csv1_path=file_set.get('csv1'); csv2_path=file_set.get('csv2'); video_load_success = False
        if video_path: video_load_success = self.playback_manager.load_video(video_path)
        else: self.ui_manager.show_message_box("critical", "Load Error", "Video path missing in file set.")
        csv_load_ok = False
        if csv1_path and os.path.exists(csv1_path):
            if self.csv_handler.set_input_file(csv1_path):
                csv_load_ok = True
                if csv2_path:
                     if os.path.exists(csv2_path):
                         if not self.csv_handler.set_second_input_file(csv2_path): self.ui_manager.show_message_box("warning", "Load Warning", f"Load CSV2 failed:\n{csv2_path}")
                     else: print(f"Controller INFO: CSV2 path provided but not found: {csv2_path}"); self.csv_handler.second_data = None
                else: self.csv_handler.second_data = None
            else: self.ui_manager.show_message_box("critical", "Load Error", f"Load CSV1 failed:\n{csv1_path}")
        else: self.ui_manager.show_message_box("critical", "Load Error", f"CSV1 not found:\n{csv1_path or 'N/A'}")
        if not csv_load_ok: return False
        props = self.playback_manager.get_video_properties() or {'total_frames': 0, 'width': 0, 'fps': 0}
        if self.window.timeline_widget: self.window.timeline_widget.set_video_properties(props.get('total_frames', 0), props.get('width', 0), props.get('fps', 0))
        if not video_load_success:
             self.ui_manager.show_message_box("warning", "Video Load Warning", "Video file failed to load or has invalid properties. Playback disabled. Audio analysis will be attempted.")
             self.ui_manager.enable_play_pause_button(False); print(f"Controller: Video load failed, starting Whisper processing."); self.processing_manager.start_whisper_processing(video_path); return True
        else:
             self.ui_manager.update_frame_info(0, props.get('total_frames', 0)); print(f"Controller: File set loaded. Starting Whisper processing."); self.processing_manager.start_whisper_processing(video_path); return True
    def load_first_file(self): # (Unchanged)
        # Start timing batch processing
        self.batch_start_time = time.time()
        self.failed_files = []  # Reset failed files list
        file_set=self.batch_processor.load_first_file(); self.load_batch_file_set(file_set) if file_set else self.ui_manager.show_message_box("warning", "No Files", "No files in batch.")
    def load_next_file(self): # (Unchanged)
        file_set=self.batch_processor.load_next_file(); self.load_batch_file_set(file_set) if file_set else None
    def load_previous_file(self): # (Unchanged)
        file_set=self.batch_processor.load_previous_file(); self.load_batch_file_set(file_set) if file_set else None
    @pyqtSlot(bool)
    def _toggle_playback(self, should_play): # (Unchanged)
        if should_play:
            blocked = False
            if not self.whisper_processed:
                 print(f"Controller STUCK_GUI_DEBUG: Play ignored - Whisper Processed={self.whisper_processed}")
                 blocked=True
            if self.waiting_for_confirmation or self.waiting_for_near_miss_assignment:
                 print(f"Controller STUCK_GUI_DEBUG: Play ignored - Waiting flags: Confirm={self.waiting_for_confirmation}, NM={self.waiting_for_near_miss_assignment}")
                 blocked=True
            if self.playback_manager.total_frames <= 0:
                print(f"Controller STUCK_GUI_DEBUG: Play ignored - Total Frames={self.playback_manager.total_frames}")
                blocked=True
            if self.playback_manager._is_paused_for_prompt: # Also check the playback manager flag
                 print(f"Controller STUCK_GUI_DEBUG: Play ignored - Player Paused For Prompt={self.playback_manager._is_paused_for_prompt}")
                 blocked=True
            if blocked:
                self.ui_manager.show_message_box("info", "Playback Blocked", "Cannot play now. Check console logs (STUCK_GUI_DEBUG) for reason.")
                self.ui_manager.set_play_button_state(False) # Ensure button shows "Play"
                return
            self.playback_manager.play()
        else:
            self.playback_manager.pause()
    @pyqtSlot(int, QImage)
    def _handle_frame_update(self, frame_number, qimage): # (Unchanged)
        if not self.window or not self.ui_manager: return
        self.ui_manager.update_frame_info(frame_number, self.playback_manager.total_frames)
        action_or_none = self.action_tracker.get_action_for_frame(frame_number)
        display_value = self._get_action_for_display(frame_number, action_or_none)
        self.ui_manager.update_action_display(display_value)
        if qimage and not qimage.isNull():
            pixmap = QPixmap.fromImage(qimage)
        if self.playback_manager.is_playing:
            self._check_timeline_triggers(frame_number)
    def _get_action_for_display(self, frame_number, current_action_code=None): # (Unchanged)
        display_value = None
        if self.waiting_for_confirmation and self.current_confirmation_info:
            phrase = self.current_confirmation_info.get('trigger_phrase','?')
            display_value = f"Confirm: {phrase} ?"
        elif self.waiting_for_near_miss_assignment and self.current_near_miss_info:
            event_data = self.current_near_miss_info.get('event', {})
            text = event_data.get('transcribed_text','?')
            display_value = f"Possible: '{text}' ?"
        elif self.pending_action_for_creation:
             action_text = config.ACTION_MAPPINGS.get(self.pending_action_for_creation, self.pending_action_for_creation)
             display_value = f"Pending: {action_text} [{self.pending_action_for_creation}]. Drag on timeline."
        elif self.current_selected_index != -1 and self.selected_timeline_range_data:
            selected_range = self.selected_timeline_range_data
            sel_code = selected_range.get('action', 'Unknown')
            sel_status = selected_range.get('status')
            if sel_code == "TBC" or sel_code == "NM":
                display_value = "Status: ??" # Display TBC/NM as ??
            elif sel_status == 'confirm_needed':
                display_value = f"Confirm: {sel_code} ?" # Indicate confirmation needed for selection
            else:
                display_value = sel_code # Show the normal code
        elif not self.whisper_processed:
            status_text = "Status: Processing Audio..."
            if self.window and self.window.shared_action_display_label:
                current_label_text = self.window.shared_action_display_label.text()
                if current_label_text.startswith("Status:"): status_text = current_label_text
            display_value = status_text
        elif current_action_code is not None and current_action_code not in ["TBC", "NM"]:
            display_value = current_action_code
        else:
            display_value = None # Will default to "Uncoded" in UIManager
        return display_value

    # (_check_timeline_triggers, _reset_timeline_iterator, _handle_video_finished, _trigger_next_confirmation, _show_confirmation_ui_internal, _trigger_near_miss_prompt, _show_near_miss_ui_internal, _play_audio_segment, _cleanup_active_snippet, _handle_discard_confirmation_prompt, _handle_discard_near_miss_prompt, _clear_prompt_state_and_resume, _calculate_action_end_frame, _force_ui_update, _apply_action - unchanged)
    def _check_timeline_triggers(self, frame_number): # (Unchanged)
        if not self.whisper_processed or not self.playback_manager.is_playing or self.waiting_for_confirmation or self.waiting_for_near_miss_assignment or not self.final_timeline_events: return
        while self.next_timeline_event_index < len(self.final_timeline_events):
            next_event = self.final_timeline_events[self.next_timeline_event_index]; trigger_frame = next_event.get('end_frame')
            if trigger_frame is None: self.next_timeline_event_index += 1; continue
            if frame_number >= trigger_frame:
                event_to_trigger = self.final_timeline_events[self.next_timeline_event_index]; self.next_timeline_event_index += 1; prompt_shown = False
                if event_to_trigger['type'] == 'command':
                     score = event_to_trigger.get('score', 0)
                     trigger_start_frame = event_to_trigger.get('start_frame')
                     event_code = event_to_trigger.get('code')
                     found_match = None
                     for r in self.action_tracker.action_ranges:
                         if (r.get('action') == event_code and
                             abs(r.get('start') - trigger_start_frame) <= 1 and
                             r.get('status') == 'confirm_needed'):
                             found_match = r.copy(); break
                     if found_match: prompt_shown = self._trigger_next_confirmation(found_match)
                elif event_to_trigger['type'] == 'near_miss':
                    trigger_start_frame = event_to_trigger.get('start_frame')
                    found_nm_range = None
                    for r in self.action_tracker.action_ranges:
                        if r.get('action') == 'NM' and r.get('start') == trigger_start_frame:
                            found_nm_range = r.copy(); break
                    if found_nm_range: prompt_shown = self._trigger_near_miss_prompt(event_to_trigger, found_nm_range)
                    else: print(f"Controller WARN: Could not find NM range placeholder for near miss event at F{trigger_start_frame}")
                if prompt_shown: break
            else: break
    def _reset_timeline_iterator(self, frame): # (Unchanged)
         self.next_timeline_event_index = 0
         if self.whisper_processed and self.final_timeline_events:
             for i, event in enumerate(self.final_timeline_events):
                 event_end_frame = event.get('end_frame')
                 if event_end_frame is not None and event_end_frame >= frame: self.next_timeline_event_index = i; break
             else: self.next_timeline_event_index = len(self.final_timeline_events)
    @pyqtSlot()
    def _handle_video_finished(self): # (Unchanged)
        print("Controller: Video finished.");
        self.waiting_for_confirmation = False; self.current_confirmation_info = None
        self.waiting_for_near_miss_assignment = False; self.current_near_miss_info = None
        self._cleanup_active_snippet()
        if self.ui_manager:
            self.ui_manager.show_discard_confirmation_button(False)
            self.ui_manager.show_discard_near_miss_button(False)
            self.ui_manager.enable_action_buttons(True, context='idle') # Reset buttons
        self.next_timeline_event_index = 0;
        self._force_ui_update(self.playback_manager.current_frame)
    def _trigger_next_confirmation(self, action_range_info): # (Unchanged)
        if not self.ui_manager or not self.playback_manager: return False;
        trigger_seek_frame = action_range_info.get('start')
        if trigger_seek_frame is None: print("Controller WARN: Cannot trigger confirmation, missing start frame in range info."); return False
        self.playback_manager.pause(triggered_by_prompt=True); self.playback_manager.seek(trigger_seek_frame);
        self.waiting_for_confirmation = True; self.current_confirmation_info = action_range_info;
        self.waiting_for_near_miss_assignment = False; self.current_near_miss_info = None # Ensure NM state is off
        print(f"Controller: Triggering confirmation for {action_range_info.get('action')}?");
        QTimer.singleShot(config.PAUSE_SETTLE_DELAY_MS + 50, lambda: self._show_confirmation_ui_internal(action_range_info)); return True
    def _show_confirmation_ui_internal(self, range_info): # (Unchanged)
         if not self.waiting_for_confirmation: return;
         phrase = range_info.get('trigger_phrase', '?');
         prompt_text = f"Confirm: {phrase} ?"
         print(f"Controller: Showing Confirmation UI: {prompt_text}")
         self.ui_manager.update_action_display(prompt_text);
         self.ui_manager.show_discard_confirmation_button(True)
         self.ui_manager.show_discard_near_miss_button(False) # Ensure other is hidden
         self.ui_manager.enable_action_buttons(True, context='confirm') # Enable all action buttons
         start_time = range_info.get('original_event_start_time', range_info.get('trigger_start'))
         end_time = range_info.get('original_event_end_time', range_info.get('trigger_end'))
         if start_time is not None and end_time is not None:
              self._play_audio_segment(start_time, end_time)
         else:
             print("Controller WARN: Missing time data for confirmation audio snippet.")
    def _trigger_near_miss_prompt(self, near_miss_event, near_miss_range): # (Unchanged)
        if not self.ui_manager or not self.playback_manager: return False;
        start_frame = near_miss_event.get('start_frame');
        if start_frame is None: return False
        self.playback_manager.pause(triggered_by_prompt=True); self.playback_manager.seek(start_frame);
        self.waiting_for_near_miss_assignment = True;
        self.current_near_miss_info = {'event': near_miss_event, 'range': near_miss_range}
        self.waiting_for_confirmation = False; self.current_confirmation_info = None # Ensure confirm state is off
        print(f"Controller: Triggering near miss prompt for '{near_miss_event.get('transcribed_text','?')}' corresponding to NM range at F{start_frame}");
        QTimer.singleShot(config.PAUSE_SETTLE_DELAY_MS + 50, lambda: self._show_near_miss_ui_internal(near_miss_event)); return True
    def _show_near_miss_ui_internal(self, event_info): # (Unchanged)
        if not self.waiting_for_near_miss_assignment: return;
        text = event_info.get('transcribed_text', '?');
        prompt_text = f"Possible: '{text}' ?";
        print(f"Controller: Showing Near Miss UI: {prompt_text}")
        self.ui_manager.update_action_display(prompt_text);
        self.ui_manager.show_discard_near_miss_button(True)
        self.ui_manager.show_discard_confirmation_button(False)
        self.ui_manager.enable_action_buttons(True, context='near_miss')
        start_time = event_info.get('start_time'); end_time = event_info.get('end_time')
        if start_time is not None and end_time is not None:
            self._play_audio_segment(start_time, end_time)
        else:
            print("Controller WARN: Missing time data for near miss audio snippet.")
    def _play_audio_segment(self, start_time, end_time): # (Unchanged)
        self._cleanup_active_snippet() # Clean up any previous snippet
        if not self.ffmpeg_path: print("Controller ERROR: Cannot play audio segment - ffmpeg path not found."); return
        if start_time is None or end_time is None or end_time <= start_time: print(f"Controller WARN: Invalid time range for audio segment: {start_time}-{end_time}"); return
        handler = self.processing_manager.current_whisper_handler or self.processing_manager.previous_whisper_handler
        full_audio_path = None; temp_dir = None
        if handler:
            full_audio_path = getattr(handler, 'audio_path', None)
            temp_dir = getattr(handler, 'temp_dir_for_cleanup', None) # Get temp dir for output
        if not full_audio_path or not os.path.exists(full_audio_path): print("Controller ERROR: Cannot play audio segment - Full audio path not found or invalid."); return
        if not temp_dir or not os.path.isdir(temp_dir): print("Controller WARN: Whisper handler temp directory not found, using system temp."); temp_dir = tempfile.gettempdir()
        try:
            fd, snippet_path = tempfile.mkstemp(suffix=".wav", prefix="snippet_", dir=temp_dir); os.close(fd)
            self.active_prompt_audio_snippet_path = snippet_path # Store for cleanup
        except Exception as e: print(f"Controller ERROR: Failed to create temporary snippet file in {temp_dir}: {e}"); self.active_prompt_audio_snippet_path = None; return
        command = [ self.ffmpeg_path, '-y', '-i', full_audio_path, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', snippet_path ]
        try:
            startupinfo = None; creationflags = 0
            if sys.platform == 'win32': startupinfo = subprocess.STARTUPINFO(); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startupinfo.wShowWindow = subprocess.SW_HIDE; creationflags=subprocess.CREATE_NO_WINDOW
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5, startupinfo=startupinfo, creationflags=creationflags) # Added timeout
            if result.returncode == 0: print(f"Controller: Audio snippet extracted successfully."); self.ui_manager.play_audio_snippet(snippet_path)
            else: print(f"Controller ERROR: ffmpeg failed to extract snippet (Code {result.returncode})."); print(f"  Stderr: {result.stderr}"); self._cleanup_active_snippet() # Clean up failed snippet
        except subprocess.TimeoutExpired: print("Controller ERROR: ffmpeg command timed out."); self._cleanup_active_snippet()
        except Exception as e: print(f"Controller ERROR: Exception running ffmpeg: {e}"); self._cleanup_active_snippet()
    def _cleanup_active_snippet(self): # (Unchanged)
        if self.active_prompt_audio_snippet_path and os.path.exists(self.active_prompt_audio_snippet_path):
            try: os.remove(self.active_prompt_audio_snippet_path);
            except Exception as e: print(f"Controller WARN: Failed to clean up snippet {self.active_prompt_audio_snippet_path}: {e}")
        self.active_prompt_audio_snippet_path = None
    @pyqtSlot()
    def _handle_discard_confirmation_prompt(self): # (Unchanged)
        if not self.waiting_for_confirmation or not self.current_confirmation_info: print("Controller WARN: Discard Confirmation clicked but no confirmation pending."); return
        discarded_range_info = self.current_confirmation_info; code_to_remove = discarded_range_info.get('action'); start_frame = discarded_range_info.get('start'); end_frame = discarded_range_info.get('end')
        if start_frame is None or end_frame is None: print("Controller ERROR: Invalid range info on discard confirmation."); self._clear_prompt_state_and_resume(); return
        print(f"Controller: User discarding Confirmation '{code_to_remove}?' for F{start_frame}-{end_frame}"); range_index_to_remove = -1
        for i, r in enumerate(self.action_tracker.action_ranges):
             if (r.get('start') == start_frame and r.get('end') == end_frame and r.get('action') == code_to_remove and r.get('status') == 'confirm_needed'): range_index_to_remove = i; break
        if range_index_to_remove == -1: print(f"Controller WARN: Could not find exact match in ActionTracker (needing confirmation) to remove discarded action {code_to_remove} for F{start_frame}-{end_frame}")
        else:
            self.history_manager.save_state(self.action_tracker.action_ranges, self.current_selected_index); self.action_tracker.action_ranges.pop(range_index_to_remove)
            if self.current_selected_index == range_index_to_remove: self.current_selected_index = -1; self.selected_timeline_range_data = None
            elif self.current_selected_index > range_index_to_remove: self.current_selected_index -= 1
            self.action_tracker.validate_ranges(force_signal_check=True, original_state_before_load=None, drag_info=None); # self._update_timeline_widget_ranges() # Let signal handle
        self._clear_prompt_state_and_resume()
    @pyqtSlot()
    def _handle_discard_near_miss_prompt(self): # (Unchanged)
        if not self.waiting_for_near_miss_assignment or not self.current_near_miss_info: print("Controller WARN: Discard Near Miss clicked but no near miss pending."); return
        near_miss_range_to_remove = self.current_near_miss_info.get('range'); near_miss_event = self.current_near_miss_info.get('event')
        if not near_miss_range_to_remove or not near_miss_event: print("Controller ERROR: Invalid near miss info on discard (missing range or event)."); self._clear_prompt_state_and_resume(); return
        start_frame = near_miss_range_to_remove.get('start')
        if start_frame is None : print("Controller ERROR: Invalid NM range info on discard (missing start frame)."); self._clear_prompt_state_and_resume(); return
        text = near_miss_event.get('transcribed_text','?')
        print(f"Controller: User discarding Near Miss '{text}' (removing NM range at F{start_frame})"); range_index_to_remove = -1
        for i, r in enumerate(self.action_tracker.action_ranges):
             if r.get('action') == 'NM' and r.get('start') == start_frame: range_index_to_remove = i; break
        if range_index_to_remove == -1: print(f"Controller WARN: Could not find NM range at F{start_frame} to remove.")
        else:
            self.history_manager.save_state(self.action_tracker.action_ranges, self.current_selected_index); self.action_tracker.action_ranges.pop(range_index_to_remove)
            if self.current_selected_index == range_index_to_remove: self.current_selected_index = -1; self.selected_timeline_range_data = None
            elif self.current_selected_index > range_index_to_remove: self.current_selected_index -= 1
            self.action_tracker.validate_ranges(force_signal_check=True, original_state_before_load=None, drag_info=None); # self._update_timeline_widget_ranges() # Let signal handle
        self._clear_prompt_state_and_resume()
    def _clear_prompt_state_and_resume(self): # (Unchanged)
        was_paused_for_prompt = self.playback_manager._is_paused_for_prompt
        print(f"Controller STUCK_GUI_DEBUG: Clearing prompt state. Flags before clear: Confirm={self.waiting_for_confirmation}, NM={self.waiting_for_near_miss_assignment}, PlayerPausedForPrompt={was_paused_for_prompt}")
        self.waiting_for_confirmation = False; self.current_confirmation_info = None
        self.waiting_for_near_miss_assignment = False; self.current_near_miss_info = None
        self._cleanup_active_snippet()
        self.pending_action_for_creation = None
        if self.ui_manager: self.ui_manager.clear_pending_action_button()
        print(f"Controller STUCK_GUI_DEBUG: Prompt state cleared. Flags after clear: Confirm={self.waiting_for_confirmation}, NM={self.waiting_for_near_miss_assignment}")
        if self.ui_manager:
            self.ui_manager.show_discard_confirmation_button(False)
            self.ui_manager.show_discard_near_miss_button(False)
            self.ui_manager.enable_action_buttons(True,
                                                 current_action=self.selected_timeline_range_data.get('action') if self.selected_timeline_range_data else None,
                                                 context='idle') # Revert context
        self._force_ui_update(self.playback_manager.current_frame);
        if was_paused_for_prompt:
            self.playback_manager.set_paused_for_prompt(False); # Ensure player knows prompt is over
            print(f"Controller STUCK_GUI_DEBUG: Resuming playback. PlayerPausedForPrompt is now: {self.playback_manager._is_paused_for_prompt}")
            self.playback_manager.play()
        else:
            print("Controller: Prompt resolved, but playback was not paused by prompt. Staying paused/stopped.")
    def _calculate_action_end_frame(self, current_event, total_video_frames): # (Unchanged)
        start_frame = current_event.get('start_frame'); min_end_frame = current_event.get('end_frame', start_frame); end_frame_for_action = total_video_frames - 1
        if start_frame is None: return 0
        if min_end_frame is None: min_end_frame = start_frame
        try:
             current_event_index = -1
             for idx, ev in enumerate(self.final_timeline_events):
                 if ev.get('start_frame') == start_frame and ev.get('type') == current_event.get('type'):
                     if ev.get('type') == 'near_miss' and ev.get('transcribed_text') != current_event.get('transcribed_text'): continue
                     current_event_index = idx; break
             if current_event_index == -1: raise ValueError("Event not found")
        except ValueError as e: print(f"WARN: Could not find current event ({current_event.get('type')}, F{start_frame}) in final_timeline_events for end frame calculation: {e}"); return max(min_end_frame, start_frame)
        next_event_start_frame = total_video_frames
        for next_idx in range(current_event_index + 1, len(self.final_timeline_events)):
             potential_next_event = self.final_timeline_events[next_idx]; next_event_start = potential_next_event.get('start_frame'); next_event_type = potential_next_event.get('type'); next_event_code = potential_next_event.get('code')
             if next_event_start is None: continue
             if (next_event_type == 'command' and next_event_code != 'STOP') or next_event_type == 'near_miss': next_event_start_frame = next_event_start; break
             elif next_event_type == 'command' and next_event_code == 'STOP': next_event_start_frame = next_event_start; break
        potential_end_frame = next_event_start_frame - 1; end_frame_for_action = max(min_end_frame, potential_end_frame); end_frame_for_action = min(end_frame_for_action, total_video_frames - 1); end_frame_for_action = max(start_frame, end_frame_for_action); return end_frame_for_action
    def _force_ui_update(self, frame_number): # (Unchanged)
        if self.ui_manager:
            action_at_frame = self.action_tracker.get_action_for_frame(frame_number)
            display_value = self._get_action_for_display(frame_number, action_at_frame)
            self.ui_manager.update_action_display(display_value);
            self.ui_manager.update_frame_info(frame_number, self.playback_manager.total_frames)
    def _apply_action(self, code, start_frame, end_frame): # (Unchanged)
        if not all(isinstance(f, int) for f in [start_frame, end_frame]): print(f"Ctrl ERROR: Invalid frame type for action '{code}'. Start={start_frame}, End={end_frame}"); return
        if start_frame < 0 or end_frame < start_frame: print(f"Ctrl ERROR: Invalid frame range F{start_frame}-F{end_frame} for action '{code}'."); return
        new_range = {'action': code, 'start': start_frame, 'end': end_frame, 'status': None};
        print(f"Controller (Internal _apply_action): Adding range {new_range}.");
        self.action_tracker.action_ranges.append(new_range);
        self.action_tracker.validate_ranges(force_signal_check=True, original_state_before_load=None, drag_info=None)

    # --- _handle_clear_manual_annotations (Unchanged from previous update) ---
    @pyqtSlot()
    def _handle_clear_manual_annotations(self): # (Unchanged)
         print("DEBUG: _handle_clear_manual_annotations called.") # DEBUG
         if native_dialogs.ask_yes_no("Confirm Clear", "Are you sure you want to remove ALL annotations for this file?", default_yes=False):
             if not self.action_tracker.action_ranges:
                 print("Controller: Clear All requested, but no annotations to clear.")
                 return
             print("Controller: Clearing all annotations.")
             self.history_manager.save_state(self.action_tracker.action_ranges, self.current_selected_index)
             self.action_tracker.clear_actions();
             print(f"DEBUG: ActionTracker ranges after clear call: {self.action_tracker.action_ranges}") # DEBUG
             self.current_selected_index = -1; self.selected_timeline_range_data = None
             if self.window and self.window.timeline_widget:
                 print("DEBUG: Telling TimelineWidget to deselect.") # DEBUG
                 self.window.timeline_widget.set_selected_range_by_index(-1)
             self._force_ui_update(self.playback_manager.current_frame);
             if self.ui_manager:
                 self.ui_manager.show_discard_confirmation_button(False);
                 self.ui_manager.show_discard_near_miss_button(False)
                 self.pending_action_for_creation = None
                 self.ui_manager.clear_pending_action_button()
             print("DEBUG: _handle_clear_manual_annotations finished (expecting signal handler).") # DEBUG


    # (_handle_timeline_ranges_edited, _handle_timeline_delete_request, _handle_timeline_range_selected, _handle_main_action_button_clicked - unchanged from previous updates)
    @pyqtSlot(list, int, str)
    def _handle_timeline_ranges_edited(self, edited_ranges_list, dragged_index, drag_type): # (Unchanged)
        print(f"Controller: Received ranges_edited signal (Dragged Index: {dragged_index}, Type: '{drag_type}')...")
        current_tracker_ranges_copy = copy.deepcopy(self.action_tracker.action_ranges); edited_ranges_copy = copy.deepcopy(edited_ranges_list)
        applied_pending = False
        if drag_type == "create" and self.pending_action_for_creation is not None and dragged_index != -1:
            if 0 <= dragged_index < len(edited_ranges_copy):
                new_range = edited_ranges_copy[dragged_index]
                if new_range.get('action') == 'TBC':
                    original_action = new_range['action']
                    new_range['action'] = self.pending_action_for_creation
                    print(f"  -> Applied pending action '{self.pending_action_for_creation}' to created range at index {dragged_index} (was {original_action}).")
                    applied_pending = True
                else:
                     print(f"  -> WARN: Expected created range at index {dragged_index} to be 'TBC', but found '{new_range.get('action')}'. Pending action not applied.")
            else:
                print(f"  -> WARN: Invalid index {dragged_index} for created range. Pending action not applied.")
            self.pending_action_for_creation = None
            if self.ui_manager: self.ui_manager.clear_pending_action_button()
        if edited_ranges_copy != current_tracker_ranges_copy:
            print("  -> Change detected, saving current state before applying.")
            self.history_manager.save_state(current_tracker_ranges_copy, self.current_selected_index)
            self.action_tracker.action_ranges = edited_ranges_copy
            drag_info = (dragged_index, drag_type) if dragged_index != -1 else None
            self.action_tracker.validate_ranges(force_signal_check=True, original_state_before_load=current_tracker_ranges_copy, drag_info=drag_info)
            final_selected_index = -1
            final_selected_data = None
            if drag_type == "create" and applied_pending and 0 <= dragged_index < len(self.action_tracker.action_ranges):
                 final_selected_index = dragged_index
                 final_selected_data = self.action_tracker.action_ranges[final_selected_index]
                 print(f"  -> Controller selection updated to newly created (and coded) index {final_selected_index}")
            elif drag_type in ["start_edge", "end_edge", "move"] and 0 <= dragged_index < len(self.action_tracker.action_ranges):
                 final_selected_index = dragged_index
                 final_selected_data = self.action_tracker.action_ranges[final_selected_index]
                 print(f"  -> Controller selection updated to dragged index {final_selected_index}")
            elif dragged_index != -1:
                 print(f"  -> Controller WARN: Dragged index {dragged_index} seems invalid after validation. Deselecting.")
                 final_selected_index = -1
                 final_selected_data = None
            self.current_selected_index = final_selected_index
            self.selected_timeline_range_data = final_selected_data
            self._force_ui_update(self.playback_manager.current_frame)
            if self.window and self.window.timeline_widget:
                self.window.timeline_widget.set_selected_range_by_index(self.current_selected_index)
        else: print("  -> No change detected compared to current tracker state. Ignoring.")
    @pyqtSlot(object)
    def _handle_timeline_delete_request(self, range_to_delete): # (Unchanged)
        if not isinstance(range_to_delete, dict): return;
        start_frame = range_to_delete.get('start'); end_frame = range_to_delete.get('end'); action_code = range_to_delete.get('action');
        print(f"Controller: Received delete request for '{action_code}' F{start_frame}-{end_frame}");
        range_index_to_remove = -1;
        current_ranges_before_delete = copy.deepcopy(self.action_tracker.action_ranges)
        try: range_index_to_remove = self.action_tracker.action_ranges.index(range_to_delete)
        except ValueError: print(f"Controller WARN: Could not find exact range object to delete: {range_to_delete}"); return
        self.history_manager.save_state(current_ranges_before_delete, self.current_selected_index)
        try:
            removed_range = self.action_tracker.action_ranges.pop(range_index_to_remove)
            print(f"Controller: Removed range {removed_range} at index {range_index_to_remove} from tracker.")
            original_selection = self.current_selected_index
            if self.current_selected_index == range_index_to_remove:
                self.current_selected_index = -1
                self.selected_timeline_range_data = None
                self.waiting_for_confirmation = False; self.current_confirmation_info = None
                self.waiting_for_near_miss_assignment = False; self.current_near_miss_info = None
                if self.ui_manager:
                     self.ui_manager.show_discard_confirmation_button(False)
                     self.ui_manager.show_discard_near_miss_button(False)
            elif self.current_selected_index > range_index_to_remove:
                self.current_selected_index -= 1
                print(f"Controller: Adjusted selected index after delete to {self.current_selected_index}")
            self.action_tracker.validate_ranges(force_signal_check=True, original_state_before_load=current_ranges_before_delete, drag_info=None)
            self._handle_timeline_range_selected(self.current_selected_index != -1, self.selected_timeline_range_data)
            self._force_ui_update(self.playback_manager.current_frame)
        except IndexError: print(f"Controller ERROR: Index {range_index_to_remove} out of bounds during delete.")
        except Exception as e: print(f"Controller ERROR during range removal by index: {e}")
    @pyqtSlot(bool, object)
    def _handle_timeline_range_selected(self, is_selected, range_data): # (Unchanged)
        player_paused_flag = self.playback_manager._is_paused_for_prompt if self.playback_manager else 'N/A'
        print(f"Controller STUCK_GUI_DEBUG: _handle_timeline_range_selected - START - is_selected={is_selected}, range_action={range_data.get('action') if range_data else 'None'}, current_idx={self.current_selected_index}")
        print(f"  -> State BEFORE logic: Confirm={self.waiting_for_confirmation}, NM={self.waiting_for_near_miss_assignment}, Pending={self.pending_action_for_creation}, PlayerPaused={player_paused_flag}")
        new_index = -1; found_range_object = None
        self._cleanup_active_snippet()
        if self.window and self.window.snippet_player and self.window.snippet_player.state() == QMediaPlayer.PlayingState: self.window.snippet_player.stop()
        if is_selected and range_data:
            try: found_range_object = next(r for r in self.action_tracker.action_ranges if r == range_data); new_index = self.action_tracker.action_ranges.index(found_range_object); print(f"Controller: Range selected, index set to {new_index} by data match.")
            except (StopIteration, ValueError): print(f"Controller WARN: Selected range data not found in action_tracker.action_ranges: {range_data}"); found_range_object = None; new_index = -1
        else: print("Controller: Range deselected, index set to -1"); found_range_object = None; new_index = -1
        selection_changed = (self.current_selected_index != new_index)
        self.current_selected_index = new_index; self.selected_timeline_range_data = found_range_object
        if selection_changed and self.pending_action_for_creation:
             print("  -> Clearing pending action due to selection change.")
             self.pending_action_for_creation = None
             if self.ui_manager: self.ui_manager.clear_pending_action_button()
        is_new_selection_a_prompt = False
        if is_selected and self.selected_timeline_range_data:
            if self.selected_timeline_range_data.get('status') == 'confirm_needed' or self.selected_timeline_range_data.get('action') == 'NM':
                is_new_selection_a_prompt = True
        if not is_new_selection_a_prompt:
             if self.waiting_for_confirmation or self.waiting_for_near_miss_assignment:
                 print("Controller STUCK_GUI_DEBUG: Clearing prompt flags because a non-prompt range was selected/deselected.")
             self.waiting_for_confirmation = False; self.waiting_for_near_miss_assignment = False
             self.current_confirmation_info = None; self.current_near_miss_info = None
             if not is_selected and self.playback_manager and self.playback_manager._is_paused_for_prompt:
                  print("Controller STUCK_GUI_DEBUG: Clearing player pause flag due to range deselection.")
                  self.playback_manager.set_paused_for_prompt(False)
        show_discard_confirm = False; show_discard_nm = False; ui_context = 'idle'; enable_actions = True
        if is_selected and self.selected_timeline_range_data:
            action_code = self.selected_timeline_range_data.get('action', 'Unknown'); status = self.selected_timeline_range_data.get('status')
            can_edit = self.window.timeline_widget._editing_enabled if self.window and self.window.timeline_widget else False
            if status == 'confirm_needed':
                print(f"Controller: Selected range needs confirmation: {action_code}"); self.waiting_for_confirmation = True; self.current_confirmation_info = self.selected_timeline_range_data; show_discard_confirm = True; ui_context = 'confirm'; enable_actions = True
                start_time = self.selected_timeline_range_data.get('original_event_start_time', self.selected_timeline_range_data.get('trigger_start')); end_time = self.selected_timeline_range_data.get('original_event_end_time', self.selected_timeline_range_data.get('trigger_end')); self._play_audio_segment(start_time, end_time)
            elif action_code == 'NM':
                print(f"Controller: Selected range is Near Miss (NM)"); self.waiting_for_near_miss_assignment = True; self.current_near_miss_info = {'event': self.selected_timeline_range_data, 'range': self.selected_timeline_range_data}; show_discard_nm = True; ui_context = 'near_miss'; enable_actions = True
                start_time = self.selected_timeline_range_data.get('original_event_start_time', self.selected_timeline_range_data.get('start_time')); end_time = self.selected_timeline_range_data.get('original_event_end_time', self.selected_timeline_range_data.get('end_time')); self._play_audio_segment(start_time, end_time)
            elif action_code == 'TBC': print(f"Controller: Selected range is To Be Coded (TBC)"); enable_actions = True; ui_context = 'tbc'
            else: ui_context = 'selected'; enable_actions = True # Normal range selected
            self.ui_manager.enable_action_buttons(enable_actions, current_action=action_code, context=ui_context)
            self.ui_manager.enable_delete_button(can_edit)
        else: # Deselected
            self.ui_manager.enable_action_buttons(True, context='idle'); self.ui_manager.enable_delete_button(False)
        self.ui_manager.show_discard_confirmation_button(show_discard_confirm); self.ui_manager.show_discard_near_miss_button(show_discard_nm)
        self._force_ui_update(self.playback_manager.current_frame)
        player_paused_flag_after = self.playback_manager._is_paused_for_prompt if self.playback_manager else 'N/A'
        print(f"Controller STUCK_GUI_DEBUG: _handle_timeline_range_selected - END - State AFTER logic: Confirm={self.waiting_for_confirmation}, NM={self.waiting_for_near_miss_assignment}, Pending={self.pending_action_for_creation}, PlayerPaused={player_paused_flag_after}")
    @pyqtSlot(str)
    def _handle_main_action_button_clicked(self, action_code): # (Unchanged)
        print(f"Controller: Main action button '{action_code}' clicked.")
        if self.pending_action_for_creation and self.pending_action_for_creation != action_code:
             print(f"  -> Clearing previous pending action '{self.pending_action_for_creation}'.")
             self.pending_action_for_creation = None
             if self.ui_manager: self.ui_manager.clear_pending_action_button()
        if self.waiting_for_confirmation and self.current_confirmation_info:
            confirmed_range_info = self.current_confirmation_info; original_code = confirmed_range_info.get('action'); start_frame = confirmed_range_info.get('start'); end_frame = confirmed_range_info.get('end')
            if start_frame is None or end_frame is None: print("Controller ERROR: Invalid range info on confirm via button click."); self._clear_prompt_state_and_resume(); return
            print(f"Controller: User confirmed '{action_code}' (Original: '{original_code}') for F{start_frame}-{end_frame} via button click."); found_range_to_update = None
            self.history_manager.save_state(self.action_tracker.action_ranges, self.current_selected_index); range_index_updated = -1
            for i, r in enumerate(self.action_tracker.action_ranges):
                 if (r.get('start') == start_frame and r.get('end') == end_frame and r.get('action') == original_code and r.get('status') == 'confirm_needed'):
                      r['action'] = action_code; r['status'] = None; found_range_to_update = r; range_index_updated = i; break
            if not found_range_to_update: print(f"Controller WARN: Could not find exact match in ActionTracker needing confirmation to update: {confirmed_range_info}")
            else:
                 if self.current_selected_index == range_index_updated: self.selected_timeline_range_data = found_range_to_update
                 self.action_tracker.validate_ranges(force_signal_check=True, original_state_before_load=None, drag_info=None); # self._update_timeline_widget_ranges() # Let signal handle
            self._clear_prompt_state_and_resume(); return
        elif self.waiting_for_near_miss_assignment and self.current_near_miss_info:
            near_miss_range_to_update_data = self.current_near_miss_info.get('range'); near_miss_event = self.current_near_miss_info.get('event')
            if not near_miss_range_to_update_data or not near_miss_event: print("Controller ERROR: Invalid near miss info on assign via button click (missing range or event)."); self._clear_prompt_state_and_resume(); return
            start_frame = near_miss_range_to_update_data.get('start'); end_frame = near_miss_range_to_update_data.get('end')
            if start_frame is None or end_frame is None: print("Controller ERROR: Invalid NM range info on assign via button click."); self._clear_prompt_state_and_resume(); return
            print(f"Controller: User assigned '{action_code}' to near miss event '{near_miss_event.get('transcribed_text', '?')}', updating NM range F{start_frame}-{end_frame} via button click."); found_range_object = None; found_range_index = -1
            for i, r in enumerate(self.action_tracker.action_ranges):
                 if r.get('action') == 'NM' and r.get('start') == start_frame: found_range_object = r; found_range_index = i; break
            if found_range_object is None: print(f"Controller WARN: Could not find NM range at F{start_frame} to update.")
            else:
                self.history_manager.save_state(self.action_tracker.action_ranges, self.current_selected_index)
                found_range_object['action'] = action_code; found_range_object['trigger_phrase'] = near_miss_event.get('transcribed_text'); found_range_object['confidence_score'] = near_miss_event.get('score'); found_range_object['status'] = None
                if self.current_selected_index == found_range_index: self.selected_timeline_range_data = found_range_object
                self.action_tracker.validate_ranges(force_signal_check=True, original_state_before_load=None, drag_info=None); # self._update_timeline_widget_ranges() # Let signal handle
            self._clear_prompt_state_and_resume(); return
        elif self.selected_timeline_range_data and self.current_selected_index != -1:
            range_to_update_index = self.current_selected_index
            if not (0 <= range_to_update_index < len(self.action_tracker.action_ranges)): print(f"Controller ERROR: Selected index {range_to_update_index} is invalid for action button click."); self.current_selected_index = -1; self.selected_timeline_range_data = None; self._handle_timeline_range_selected(False, None); return
            range_to_update = self.action_tracker.action_ranges[range_to_update_index]; original_code = range_to_update.get('action'); start_frame = range_to_update.get('start'); end_frame = range_to_update.get('end'); needs_update = False
            if action_code != original_code: needs_update = True
            elif range_to_update.get('status') is not None: needs_update = True
            elif original_code == 'TBC': needs_update = True
            if needs_update:
                 self.history_manager.save_state(self.action_tracker.action_ranges, self.current_selected_index); print(f"Controller: Changing selected range F{start_frame}-{end_frame} (Index {range_to_update_index}) from '{original_code}'/'{range_to_update.get('status')}' to '{action_code}'/None")
                 range_to_update['action'] = action_code; range_to_update['status'] = None; self.action_tracker.validate_ranges(force_signal_check=True, original_state_before_load=None, drag_info=None); # self._update_timeline_widget_ranges(); # Let signal handle
                 self._force_ui_update(self.playback_manager.current_frame); self.ui_manager.enable_action_buttons(True, current_action=action_code, context='selected')
            else: print(f"Controller: Action '{action_code}' already assigned to selected range. No change.")
        else:
            print(f"Controller: No range selected and no prompt active. Setting pending action for creation: '{action_code}'")
            self.pending_action_for_creation = action_code
            if self.ui_manager:
                self.ui_manager.set_pending_action_button(action_code)
                self._force_ui_update(self.playback_manager.current_frame) # Update status display

    @pyqtSlot()
    def _update_timeline_widget_ranges(self):
        print("DEBUG: _update_timeline_widget_ranges called.") # DEBUG
        if not self.window or not self.window.timeline_widget:
            print("DEBUG: _update_timeline_widget_ranges - No window/timeline widget.") # DEBUG
            return
        current_ranges_from_tracker = copy.deepcopy(self.action_tracker.action_ranges)
        print(f"DEBUG: Calling timeline_widget.update_ranges with {len(current_ranges_from_tracker)} ranges.") # DEBUG
        self.window.timeline_widget.update_ranges(current_ranges_from_tracker)
        has_ranges = bool(current_ranges_from_tracker);
        can_edit = self.window.timeline_widget._editing_enabled if self.window and self.window.timeline_widget else False
        if self.ui_manager:
            self.ui_manager.enable_clear_button(has_ranges and can_edit)
            self.ui_manager.enable_delete_button(self.current_selected_index != -1 and can_edit)
        if self.playback_manager and not self.playback_manager.is_playing:
             self._force_ui_update(self.playback_manager.current_frame)
        print("DEBUG: _update_timeline_widget_ranges finished.") # DEBUG


    @pyqtSlot(list)
    def _handle_whisper_results(self, segments):
        print(f"Controller: Received Whisper results via ProcessingManager ({len(segments)} segments).")
        self.whisper_processed = True
        self.final_timeline_events = []
        self.generated_action_ranges = []

        props = self.playback_manager.get_video_properties()
        fps = props.get('fps')
        total_frames = props.get('total_frames')

        if not fps or fps <= 0 or not total_frames or total_frames <= 0:
             error_msg = f"Cannot process Whisper results: Invalid video properties (FPS: {fps}, Frames: {total_frames})."
             self.ui_manager.update_action_display(f"Status: Error - Invalid Video Props")
             if self.ui_manager:
                 self.ui_manager.show_message_box("critical", "Processing Error", error_msg)
                 self.ui_manager.enable_play_pause_button(False)
             # Enable editing even on error, but don't proceed with range generation
             if self.window.timeline_widget:
                 self.window.timeline_widget.set_video_properties(total_frames or 0, props.get('width', 0), fps or 0)
                 self.window.timeline_widget.set_editing_enabled(True)
                 print("Controller: Timeline editing ENABLED (Whisper error, invalid props).")
             if self.ui_manager:
                 self.ui_manager.show_discard_confirmation_button(False)
                 self.ui_manager.show_discard_near_miss_button(False)
             self._update_timeline_widget_ranges() # Ensure timeline redraws if empty
             return

        if self.window.timeline_widget:
             self.window.timeline_widget.set_video_properties(total_frames, props.get('width', 0), fps)

        print(f"Controller: Calling TimelineProcessor...")
        self.final_timeline_events, self.generated_action_ranges = self.timeline_processor.process_segments( segments, fps, total_frames )
        print(f"Controller: TimelineProcessor finished. Events: {len(self.final_timeline_events)}, Ranges: {len(self.generated_action_ranges)}")

        # --- MOVED: Enable timeline editing EARLIER ---
        if self.window.timeline_widget:
            self.window.timeline_widget.set_editing_enabled(True)
            print("Controller: Timeline editing ENABLED (before validation).")
        # --- END MOVED ---

        print("Controller: Applying initial generated ranges to ActionTracker...")
        initial_ranges_to_add = []
        if self.generated_action_ranges:
            for r in sorted(self.generated_action_ranges, key=lambda x: x.get('start_frame', 0)):
                if r.get('start_frame') is not None and r.get('end_frame') is not None:
                    # Extract only relevant keys for ActionTracker ranges
                    range_data = { k: v for k, v in r.items() if k in [
                        'action_code', 'start_frame', 'end_frame',
                        'start_time', 'end_time', 'trigger_phrase',
                        'trigger_start', 'trigger_end', 'trigger_start_frame',
                        'trigger_end_frame', 'confidence_score', 'status',
                        'matched_words', 'original_event_start_time',
                        'original_event_end_time', 'transcribed_text' # Include for NM
                    ]}
                    # Rename keys for ActionTracker standard
                    range_data['action'] = range_data.pop('action_code')
                    range_data['start'] = range_data.pop('start_frame')
                    range_data['end'] = range_data.pop('end_frame')
                    initial_ranges_to_add.append(range_data)

        if initial_ranges_to_add:
            # Save state BEFORE adding whisper ranges if existing ranges exist
            if self.action_tracker.action_ranges:
                 self.history_manager.save_state(self.action_tracker.action_ranges, self.current_selected_index)

            # Add new ranges
            self.action_tracker.action_ranges.extend(initial_ranges_to_add)
            print(f"Controller: Extended action_tracker.ranges. Length before validate: {len(self.action_tracker.action_ranges)}")

            # Validate ranges (this will emit action_ranges_changed if changes occur)
            self.action_tracker.validate_ranges(force_signal_check=True, original_state_before_load=None, drag_info=None)
            print(f"Controller: Called validate_ranges with force_signal_check after extending.")
            # self._update_timeline_widget_ranges() # Let signal from validate_ranges handle this update

        elif not self.action_tracker.action_ranges:
             # No ranges generated and tracker was empty, ensure timeline is updated
             print("Controller: No ranges generated and tracker empty, calling _update_timeline_widget_ranges directly.")
             self._update_timeline_widget_ranges()
        else:
             # No ranges generated, but tracker had existing ranges. Validate just in case.
             print("Controller: No command ranges generated, existing ranges remain. Calling validate just in case.")
             original_state = copy.deepcopy(self.action_tracker.action_ranges)
             self.action_tracker.validate_ranges(original_state_before_load=original_state, drag_info=None) # Let signal handle update if needed

        self.next_timeline_event_index = 0

        # Enable UI elements now that processing is done
        if self.ui_manager:
            self.ui_manager.enable_play_pause_button(True)
            self.ui_manager.enable_action_buttons(True) # Enable action buttons
            self.ui_manager.show_discard_confirmation_button(False) # Ensure prompts hidden
            self.ui_manager.show_discard_near_miss_button(False)

        # Force a final UI refresh for frame 0
        self._force_ui_update(0)
        print(f"Controller: Whisper processing complete.")


    # (_handle_processing_error, _initiate_save, _handle_save_complete, cleanup_on_exit, _handle_processing_status_update, _handle_delete_selected_range_button, _do_undo, _do_redo, _apply_restored_state, _update_undo_redo_buttons - unchanged)
    @pyqtSlot(str, str)
    def _handle_processing_error(self, error_type, message): # (Unchanged)
        print(f"Controller ERROR ({error_type}): {message}"); title = "Whisper Error" if error_type == "whisper" else "Processing Error"; status_msg = f"Status: Error - {title}"
        if self.ui_manager: self.ui_manager.show_message_box("critical", title, message); self.ui_manager.update_action_display(status_msg)
        if error_type == "whisper":
             if self.window.timeline_widget: self.window.timeline_widget.set_editing_enabled(True); print("Controller: Timeline editing ENABLED (despite Whisper error).")
             if self.ui_manager: self.ui_manager.enable_play_pause_button(self.playback_manager.total_frames > 0); self.ui_manager.enable_action_buttons(True); self.ui_manager.enable_delete_button(False)
             self.whisper_processed = False;
             if self.ui_manager: self.ui_manager.show_discard_confirmation_button(False); self.ui_manager.show_discard_near_miss_button(False)
             self._update_timeline_widget_ranges()
    @pyqtSlot()
    def _initiate_save(self): # (Unchanged)
        print(f"--- Initiate Save Triggered ---")
        active_file_set = self.current_file_set
        if active_file_set is None and self.batch_processor.get_current_index() != -1: active_file_set = self.batch_processor.get_current_file_set()
        if not active_file_set:
             if self.ui_manager: self.ui_manager.show_message_box("warning", "No File", "No file available to save."); return
        if self.processing_manager.save_thread and self.processing_manager.save_thread.isRunning():
             if self.ui_manager: self.ui_manager.show_message_box("warning", "Busy", "Already saving outputs."); return
        if self.processing_manager.whisper_thread and self.processing_manager.whisper_thread.isRunning():
             if self.ui_manager: self.ui_manager.show_message_box("warning", "Busy", "Whisper processing is still running. Please wait or cancel."); return
        if self.waiting_for_confirmation or self.waiting_for_near_miss_assignment:
             if native_dialogs.ask_yes_no("Resolve Prompt?", "A confirmation or near miss assignment is pending.\nDiscard the prompt to continue saving?", default_yes=False):
                 print("Controller: Discarding active prompt to allow saving.")
                 if self.waiting_for_confirmation: self._handle_discard_confirmation_prompt()
                 elif self.waiting_for_near_miss_assignment: self._handle_discard_near_miss_prompt()
                 if self.waiting_for_confirmation or self.waiting_for_near_miss_assignment: print("Controller ERROR: Failed to clear prompt state before saving."); self.ui_manager.show_message_box("warning", "Save Error", "Failed to clear pending prompt. Cannot save yet."); return
             else:
                 native_dialogs.show_info("Save Cancelled", "Save cancelled. Please resolve the prompt first.")
                 return
        placeholders_exist = any(r.get('action') in ['TBC', 'NM'] or r.get('status') == 'confirm_needed' for r in self.action_tracker.action_ranges)
        if placeholders_exist:
            if not native_dialogs.ask_yes_no("Unresolved Ranges", "There are unassigned ('??') or unconfirmed ('CODE?') ranges.\nThese will be saved as '' (blank).\n\nDo you want to continue saving?", default_yes=False):
                return
        self.playback_manager.pause(); last_frame = self.playback_manager.total_frames - 1 if self.playback_manager.total_frames > 0 else 0
        print(f"Controller: Validating tracker for save (up to F{last_frame})..."); self.action_tracker.set_frame(last_frame); self.action_tracker.validate_ranges(drag_info=None)
        actions_dict, num_frames_in_dict = self.action_tracker.get_all_actions_for_export()
        print(f"Controller: Exporting {len(actions_dict)} action entries for {num_frames_in_dict} frames (after filtering placeholders).")
        actual_actions_count = sum(1 for action in actions_dict.values() if action not in ["", "BL"]) # Check for non-blank, non-BL
        if actual_actions_count == 0 and num_frames_in_dict > 0:
            if not native_dialogs.ask_yes_no("No Actions Coded", f"Only blank or 'BL' recorded for {num_frames_in_dict} frames (excluding placeholders).\nSave blank/BL-only files?", default_yes=False):
                return
        elif num_frames_in_dict == 0:
             if not native_dialogs.ask_yes_no("No Data", "No frames found in export data (after filtering placeholders).\nAttempt save empty/blank files anyway?", default_yes=False):
                 return
        try:
            video_path = active_file_set.get('video'); csv1_path = active_file_set.get('csv1'); csv2_path = active_file_set.get('csv2')
            if not video_path or not csv1_path : raise ValueError("Missing video or primary CSV path in active_file_set.")
            current_video_dir = os.path.dirname(video_path); batch_parent_dir = os.path.dirname(current_video_dir)
            if not batch_parent_dir or batch_parent_dir == current_video_dir: batch_parent_dir = current_video_dir
            # Always use standard output location from config_paths
            output_dir = str(config_paths.get_output_base_dir())
            os.makedirs(output_dir, exist_ok=True)
            print(f"Controller: Output directory set to: {output_dir}")
            # Store output directory for completion summary
            self.output_directory = output_dir
            video_bn = os.path.splitext(os.path.basename(video_path))[0]; csv1_bn = os.path.splitext(os.path.basename(csv1_path))[0]
            out_csv = os.path.join(output_dir, f"{csv1_bn}_coded.csv"); out_vid = os.path.join(output_dir, f"{video_bn}_coded.mp4"); out_csv2 = None
            if csv2_path: csv2_bn = os.path.splitext(os.path.basename(csv2_path))[0]; out_csv2 = os.path.join(output_dir, f"{csv2_bn}_coded.csv")
        except Exception as e:
            error_msg = f"Could not determine output paths: {e}\nTraceback:\n{traceback.format_exc()}"; print(f"Controller ERROR: {error_msg}")
            if self.ui_manager: self.ui_manager.show_message_box("critical", "Output Path Error", error_msg); return
        if self.ui_manager: self.ui_manager.show_progress_bar(True)
        self.processing_manager.start_save_processing( video_path, self.action_tracker, self.csv_handler, actions_dict, out_csv, out_vid, out_csv2 )
    @pyqtSlot(bool)
    def _handle_save_complete(self, success): # (Modified for comprehensive completion summary)
        if self.ui_manager: self.ui_manager.show_progress_bar(False)

        # Track failed files
        if not success:
            current_file_set = self.current_file_set
            if current_file_set:
                base_id = current_file_set.get('base_id', 'Unknown')
                self.failed_files.append(base_id)

            if not native_dialogs.ask_yes_no("Save Error", "Error saving file (check console/logs for details).\n\nContinue processing the next file?", default_yes=True):
                print("Controller: Batch stopped by user after save error.")
                # Show early exit summary
                self._show_batch_completion_summary(incomplete=True)
                return

        if self.batch_processor.has_next_file():
            print("Controller: Save complete (or user chose to continue), loading next file...")
            # Force garbage collection between files to prevent memory accumulation
            gc.collect()
            QTimer.singleShot(100, self.load_next_file)
        else:
            print("Controller: Save complete. Batch finished.")
            # Show comprehensive completion summary and exit
            self._show_batch_completion_summary(incomplete=False)

    def _show_batch_completion_summary(self, incomplete=False):
        """
        Show comprehensive batch processing completion summary with native dialog.
        Exits the application after user acknowledges the summary.

        Args:
            incomplete: True if batch was stopped early, False if completed normally
        """
        # Calculate processing time
        if self.batch_start_time:
            total_seconds = time.time() - self.batch_start_time
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
        else:
            time_str = "Unknown"

        # Calculate statistics
        total_files = self.batch_processor.get_total_files()
        current_index = self.batch_processor.get_current_index()
        processed_count = current_index + 1  # Index is 0-based
        failed_count = len(self.failed_files)
        successful_count = processed_count - failed_count

        # Build summary message
        if incomplete:
            summary = f"Batch Processing Stopped\n\n"
        else:
            summary = f"Batch Processing Complete!\n\n"

        summary += f"Files processed: {processed_count} of {total_files}\n"
        summary += f"  • Successful: {successful_count}\n"
        if failed_count > 0:
            summary += f"  • Failed: {failed_count}\n"
            # List failed files
            summary += f"\nFailed files:\n"
            for failed_id in self.failed_files[:5]:  # Show first 5
                summary += f"  • {failed_id}\n"
            if len(self.failed_files) > 5:
                summary += f"  ... and {len(self.failed_files) - 5} more\n"

        summary += f"\nProcessing time: {time_str}\n"

        if self.output_directory:
            summary += f"\nOutput location:\n{self.output_directory}"

        # Show native dialog
        if incomplete:
            native_dialogs.show_warning("Processing Stopped", summary)
        else:
            native_dialogs.show_info("Processing Complete", summary)

        # Disable UI and exit application
        self.ui_manager.set_batch_navigation(False, current_index, total_files)
        print("Controller: Exiting application after batch completion")
        QTimer.singleShot(100, lambda: self.app.quit())

    def cleanup_on_exit(self): # (Unchanged)
        print("Controller: Cleanup on exit..."); self._cleanup_active_snippet()
        if self.processing_manager: self.processing_manager.cleanup_on_exit()
        if self.playback_manager and self.playback_manager.player:
            print("Controller: Attempting player resource cleanup...");
            try: self.playback_manager.player.__del__(); print("Controller: Explicit player cleanup attempted.")
            except Exception as e: print(f"Controller WARN: Error during explicit player cleanup: {type(e).__name__}: {e}")
        print("Controller: Cleanup finished.")
    @pyqtSlot(str)
    def _handle_processing_status_update(self, message): # (Unchanged)
        if self.ui_manager: self.ui_manager.update_action_display(f"Status: {message}")
    @pyqtSlot()
    def _handle_delete_selected_range_button(self): # (Unchanged)
        if not self.selected_timeline_range_data: print("Controller: Delete button clicked, but no range selected."); return
        can_edit = self.window.timeline_widget._editing_enabled if self.window and self.window.timeline_widget else False
        if not can_edit: print("Controller: Delete button clicked, but editing is disabled."); return
        range_repr = f"'{self.selected_timeline_range_data.get('action')}' F{self.selected_timeline_range_data.get('start')}-{self.selected_timeline_range_data.get('end')}"
        print(f"Controller: Deleting range via button: {range_repr}"); range_to_delete_copy = copy.deepcopy(self.selected_timeline_range_data)
        self._handle_timeline_delete_request(range_to_delete_copy) # Calls separate handler
    @pyqtSlot()
    def _do_undo(self): # (Unchanged)
        print("Controller: Undo requested.");
        if not self.history_manager.can_undo(): print("Controller: Nothing to undo."); return
        current_ranges = self.action_tracker.action_ranges; current_selection = self.current_selected_index
        self.history_manager.undo(current_ranges, current_selection)
    @pyqtSlot()
    def _do_redo(self): # (Unchanged)
        print("Controller: Redo requested.");
        if not self.history_manager.can_redo(): print("Controller: Nothing to redo."); return
        current_ranges = self.action_tracker.action_ranges; current_selection = self.current_selected_index
        self.history_manager.redo(current_ranges, current_selection)
    @pyqtSlot(list, int)
    def _apply_restored_state(self, restored_ranges, restored_selection_index): # (Unchanged)
        print(f"Controller: Applying restored state (Ranges: {len(restored_ranges)}, Selected: {restored_selection_index})")
        if self.waiting_for_confirmation or self.waiting_for_near_miss_assignment or self.pending_action_for_creation:
             print("  -> Clearing active prompt/pending state due to history restore."); self.waiting_for_confirmation = False; self.current_confirmation_info = None; self.waiting_for_near_miss_assignment = False; self.current_near_miss_info = None; self._cleanup_active_snippet()
             self.pending_action_for_creation = None
             if self.ui_manager:
                 self.ui_manager.show_discard_confirmation_button(False); self.ui_manager.show_discard_near_miss_button(False);
                 self.ui_manager.clear_pending_action_button()
                 self.ui_manager.enable_action_buttons(True, context='idle')
        self.action_tracker.action_ranges = restored_ranges
        if 0 <= restored_selection_index < len(restored_ranges):
             self.current_selected_index = restored_selection_index; self.selected_timeline_range_data = restored_ranges[restored_selection_index]
             restored_range = self.selected_timeline_range_data
             if restored_range.get('status') == 'confirm_needed': print("  -> Restored selection requires confirmation."); self._handle_timeline_range_selected(True, restored_range)
             elif restored_range.get('action') == 'NM': print("  -> Restored selection is Near Miss."); self._handle_timeline_range_selected(True, restored_range)
             else: self._handle_timeline_range_selected(True, restored_range) # Make sure UI updates for normal selection restore too
        else: self.current_selected_index = -1; self.selected_timeline_range_data = None; self._handle_timeline_range_selected(False, None) # Explicitly handle deselection
        if self.window and self.window.timeline_widget: QTimer.singleShot(0, lambda idx=self.current_selected_index: self.window.timeline_widget.set_selected_range_by_index(idx) if self.window and self.window.timeline_widget else None)
        # Let signal from validate_ranges handle this if needed
        # self._update_timeline_widget_ranges();
        self._force_ui_update(self.playback_manager.current_frame); print("Controller: Finished applying restored state.")
    @pyqtSlot()
    def _update_undo_redo_buttons(self): # (Unchanged)
        if self.ui_manager:
            can_undo = self.history_manager.can_undo(); can_redo = self.history_manager.can_redo()
            self.ui_manager.enable_undo_button(can_undo); self.ui_manager.enable_redo_button(can_redo)

# --- END OF app_controller.py ---