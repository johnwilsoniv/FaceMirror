# action_tracker.py
# --- START OF FILE action_tracker.py ---

from PyQt5.QtCore import QObject, pyqtSignal
import copy # Import copy for deepcopy

class ActionTracker(QObject):
    action_ranges_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.action_ranges = []
        self.current_frame = 0
        self.active_action = None

    def set_frame(self, frame):
        self.current_frame = frame

    def start_action(self, action_code):
        # (No change)
        if self.active_action and self.action_ranges:
            self.stop_action()
        self.active_action = action_code
        self.action_ranges.append({
            'action': action_code, 'start': self.current_frame, 'end': None, 'status': None
        })

    def continue_action(self): # (No change)
        pass

    def stop_action(self): # (No change)
        action_stopped = False
        if self.active_action and self.action_ranges:
            for i in range(len(self.action_ranges)-1, -1, -1):
                if self.action_ranges[i]['action'] == self.active_action and self.action_ranges[i]['end'] is None:
                    self.action_ranges[i]['end'] = max(self.current_frame, self.action_ranges[i]['start'])
                    action_stopped = True
                    break
        self.active_action = None
        if action_stopped:
             original_state = copy.deepcopy(self.action_ranges)
             self.validate_ranges(force_signal_check=True, original_state_before_load=original_state, drag_info=None)

    def get_action_for_frame(self, frame): # (No change)
        for range_data in reversed(self.action_ranges):
            start_frame = range_data.get('start'); end_frame = range_data.get('end'); action_code = range_data.get('action')
            if start_frame is None or end_frame is None or action_code is None: continue
            if start_frame <= frame <= end_frame:
                if action_code in ["TBC", "NM"]: return None
                else: return action_code
        return None

    # --- MODIFIED get_all_actions_for_export ---
    def get_all_actions_for_export(self):
        """
        Generates a dictionary mapping frame numbers to action codes for export,
        filtering out placeholder ranges (TBC, NM) and unconfirmed ranges,
        treating uncoded areas as '' (blank). Exports valid 'BL' ranges correctly.
        """
        frame_actions = {}
        max_frame = 0

        # Filter ranges before calculating max_frame and generating dict
        exportable_ranges = [
            r for r in self.action_ranges
            if r.get('action') not in ['TBC', 'NM'] and r.get('status') != 'confirm_needed'
        ]

        if not exportable_ranges: return {}, 0 # Return empty if no valid ranges exist

        # Calculate max_frame only from valid, exportable ranges
        for range_data in exportable_ranges:
            end = range_data.get('end')
            if end is not None and isinstance(end, int) and end > max_frame: max_frame = end
            start = range_data.get('start') # Check start too in case of single-frame actions at the end
            if start is not None and isinstance(start, int) and start > max_frame: max_frame = start

        total_frame_count_for_export = max_frame + 1
        # --- MODIFICATION: Default to blank string instead of 'NC' ---
        frame_actions = {f: "" for f in range(total_frame_count_for_export)} # Default to BLANK

        # Fill dictionary using only exportable ranges (this will correctly add 'BL' if present)
        for range_data in exportable_ranges:
            start = range_data.get('start'); end = range_data.get('end'); action = range_data.get('action')
            if start is None or end is None or action is None: continue # Should already be filtered, but check again

            for frame in range(start, end + 1):
                if 0 <= frame < total_frame_count_for_export:
                    frame_actions[frame] = action # Overwrite blank default with action ('BL', 'RE', etc.)
                else:
                    # This warning indicates an issue with max_frame calculation or range data
                    print(f"ActionTracker Export WARN: Frame {frame} out of calculated bounds (0-{max_frame}). Exportable Range: {range_data}")

        # print(f"ActionTracker Export: Returning {len(frame_actions)} frames, based on {len(exportable_ranges)} exportable ranges.") # Less verbose
        return frame_actions, total_frame_count_for_export
    # --- End MODIFIED ---

    def clear_actions(self): # (No change)
        if self.action_ranges:
            print("ActionTracker: Clearing actions and emitting signal.")
            self.action_ranges = []; self.active_action = None
            self.action_ranges_changed.emit()

    def save_actions(self, file_path): # (No change)
        self.validate_ranges(drag_info=None)
        import json
        try:
            with open(file_path, 'w') as f: json.dump(self.action_ranges, f, default=int)
            print(f"ActionTracker: Saved {len(self.action_ranges)} ranges (including placeholders) to {file_path}")
            return True
        except Exception as e: print(f"Error saving actions: {e}"); return False

    def load_actions(self, file_path): # (No change)
        import json
        initial_ranges = copy.deepcopy(self.action_ranges)
        try:
            with open(file_path, 'r') as f: self.action_ranges = json.load(f)
            print(f"ActionTracker: Loaded {len(self.action_ranges)} ranges from {file_path}")
            self.validate_ranges(force_signal_check=True, original_state_before_load=initial_ranges, drag_info=None)
            self._update_timeline_widget_ranges()
            return True
        except Exception as e:
            print(f"Error loading actions: {e}"); self.action_ranges = []
            if initial_ranges: print("ActionTracker: Emitting signal after load error (cleared ranges)."); self.action_ranges_changed.emit()
            return False

    def validate_ranges(self, force_signal_check=False, original_state_before_load=None, drag_info=None): # (No change)
        original_ranges_before_validation = copy.deepcopy(self.action_ranges); made_changes_during_validation = False
        valid_ranges = []
        for i, r in enumerate(self.action_ranges):
            current_r = r.copy(); start = current_r.get('start'); end = current_r.get('end'); action = current_r.get('action')
            if start is None or action is None: print(f"Validate WARN: Removing range {i} due to missing start/action: {r}"); made_changes_during_validation = True; continue
            if end is None:
                 is_last_active_range = False
                 if self.active_action == action:
                     last_idx_for_action = -1
                     for idx, range_item in enumerate(self.action_ranges):
                         if range_item.get('action') == action: last_idx_for_action = idx
                     if i == last_idx_for_action: is_last_active_range = True
                 if is_last_active_range: end = self.current_frame; print(f"Validate INFO: Ending active action '{action}' (range {i}) at current frame {end}.")
                 else: end = start; print(f"Validate WARN: Range {i} ('{action}') had None end. Setting end=start={start}.")
                 current_r['end'] = end; made_changes_during_validation = True
            if start is not None and end is not None and start > end: print(f"Validate WARN: Range {i} ('{action}') had start ({start}) > end ({end}). Swapping."); current_r['start'], current_r['end'] = end, start; made_changes_during_validation = True
            valid_ranges.append(current_r)
        self.action_ranges = valid_ranges
        if not self.action_ranges:
             if original_ranges_before_validation: print("ActionTracker: Emitting signal because validation removed all ranges."); self.action_ranges_changed.emit()
             return made_changes_during_validation
        self.action_ranges.sort(key=lambda x: x.get('start', 0))
        resolved_ranges = []
        if self.action_ranges:
            resolved_ranges.append(copy.deepcopy(self.action_ranges[0]))
            for i in range(1, len(self.action_ranges)):
                current_range_copy = copy.deepcopy(self.action_ranges[i]); last_resolved_range = resolved_ranges[-1]
                current_start = current_range_copy.get('start'); last_resolved_end = last_resolved_range.get('end')
                if current_start is None or last_resolved_end is None: print(f"Validate WARN: Skipping overlap check due to None start/end in range {i} or previous resolved."); resolved_ranges.append(current_range_copy); continue
                if current_start <= last_resolved_end:
                    made_changes_during_validation = True; last_resolved_start = last_resolved_range.get('start')
                    is_end_drag_of_previous = False
                    if drag_info:
                        dragged_idx, drag_type = drag_info
                        if dragged_idx == (i - 1) and drag_type == "end_edge": is_end_drag_of_previous = True
                    if is_end_drag_of_previous:
                        new_start_for_current = last_resolved_range['end'] + 1
                        current_end = current_range_copy.get('end')
                        if current_end is not None and new_start_for_current <= current_end: current_range_copy['start'] = new_start_for_current; resolved_ranges.append(current_range_copy)
                        else: print(f"Validate WARN: Pushing start to {new_start_for_current} would invalidate range {i} (End: {current_end}). Keeping original current range and adding."); resolved_ranges.append(self.action_ranges[i].copy())
                    else:
                        new_end_for_previous = current_start - 1
                        if last_resolved_start is None: print(f"Validate WARN: Cannot resolve overlap, previous resolved range has None start."); resolved_ranges.append(current_range_copy); continue
                        if new_end_for_previous < last_resolved_start: print(f"  -> Range {i} ('{current_range_copy.get('action')}') engulfs previous ('{last_resolved_range.get('action')}'). Replacing previous."); resolved_ranges.pop(); resolved_ranges.append(current_range_copy)
                        else:
                            if last_resolved_range['end'] != new_end_for_previous: last_resolved_range['end'] = new_end_for_previous
                            resolved_ranges.append(current_range_copy)
                else: resolved_ranges.append(current_range_copy)
        self.action_ranges = resolved_ranges
        emit_signal = False
        if made_changes_during_validation: print("ActionTracker: Emitting signal due to changes during validation steps."); emit_signal = True
        elif force_signal_check:
            state_before_operation = original_state_before_load if original_state_before_load is not None else []
            if self.action_ranges != state_before_operation: emit_signal = True
        if emit_signal: self.action_ranges_changed.emit()
        return made_changes_during_validation

    def _update_timeline_widget_ranges(self): # (No change)
        print("ActionTracker: Explicitly emitting action_ranges_changed for timeline update.")
        self.action_ranges_changed.emit()

# --- END OF FILE action_tracker.py ---