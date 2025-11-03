# history_manager.py

from PyQt5.QtCore import QObject, pyqtSignal
import copy
import traceback # For debugging state saving issues

class HistoryManager(QObject):
    """Manages undo/redo state using snapshots."""

    # Emitted when the ability to undo/redo changes (for button enablement)
    history_changed = pyqtSignal()
    # Emitted AFTER undo/redo, carrying the state to be applied
    state_restored = pyqtSignal(list, int) # restored_ranges, restored_selection_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50 # Limit history size

    def clear(self):
        """Clears both undo and redo stacks."""
        if self.undo_stack or self.redo_stack:
            self.undo_stack = []
            self.redo_stack = []
            print("HistoryManager: Cleared history.")
            self.history_changed.emit()

    def save_state(self, current_ranges, current_selection_index):
        """
        Saves the current state to the undo stack BEFORE an action is performed.
        Clears the redo stack.
        """
        try:
            snapshot = {
                'ranges': copy.deepcopy(current_ranges),
                'selected_index': current_selection_index
            }
            # print(f"HistoryManager: Saving state (Ranges: {len(snapshot['ranges'])}, Selected: {snapshot['selected_index']})") # Verbose log

            self.undo_stack.append(snapshot)
            # Limit history size
            if len(self.undo_stack) > self.max_history:
                self.undo_stack.pop(0)

            # Performing a new action invalidates the redo history
            redo_was_cleared = False
            if self.redo_stack:
                self.redo_stack = []
                redo_was_cleared = True
                print("HistoryManager: Cleared redo stack.")

            # Emit signal if undo became possible or redo became impossible
            if len(self.undo_stack) == 1 or redo_was_cleared:
                 self.history_changed.emit()

        except Exception as e:
            print(f"ERROR in HistoryManager.save_state: {e}")
            print(f"Current Selection Index type: {type(current_selection_index)}")
            print(f"Current Ranges type: {type(current_ranges)}")
            # Add more detailed traceback if needed
            # traceback.print_exc()


    def _push_to_other_stack(self, target_stack, current_ranges, current_selection_index):
        """Internal helper to push state during undo/redo without clearing."""
        try:
            snapshot = {
                 'ranges': copy.deepcopy(current_ranges),
                 'selected_index': current_selection_index
            }
            target_stack.append(snapshot)
            # Limit history size for the target stack too
            if len(target_stack) > self.max_history:
                 target_stack.pop(0)
        except Exception as e:
            print(f"ERROR in HistoryManager._push_to_other_stack: {e}")
            # traceback.print_exc()

    def undo(self, current_ranges_before_undo, current_selection_before_undo):
        """
        Moves the current state to redo stack, restores the previous state from undo stack.
        """
        if not self.can_undo():
            print("HistoryManager: Nothing to undo.")
            return False

        # 1. Save the current state (the one we are undoing *from*) to the redo stack
        self._push_to_other_stack(self.redo_stack, current_ranges_before_undo, current_selection_before_undo)

        # 2. Pop the state we want to restore from the undo stack
        restored_state = self.undo_stack.pop()
        restored_ranges = restored_state.get('ranges', [])
        restored_index = restored_state.get('selected_index', -1)
        print(f"HistoryManager: Undoing. Restoring state (Ranges: {len(restored_ranges)}, Selected: {restored_index})")

        # 3. Emit signals
        self.state_restored.emit(restored_ranges, restored_index)
        self.history_changed.emit()
        return True

    def redo(self, current_ranges_before_redo, current_selection_before_redo):
        """
        Moves the current state to undo stack, restores the next state from redo stack.
        """
        if not self.can_redo():
            print("HistoryManager: Nothing to redo.")
            return False

        # 1. Save the current state (the one we are redoing *from*) to the undo stack
        self._push_to_other_stack(self.undo_stack, current_ranges_before_redo, current_selection_before_redo)

        # 2. Pop the state we want to restore from the redo stack
        restored_state = self.redo_stack.pop()
        restored_ranges = restored_state.get('ranges', [])
        restored_index = restored_state.get('selected_index', -1)
        print(f"HistoryManager: Redoing. Restoring state (Ranges: {len(restored_ranges)}, Selected: {restored_index})")

        # 3. Emit signals
        self.state_restored.emit(restored_ranges, restored_index)
        self.history_changed.emit()
        return True

    def can_undo(self):
        """Check if there are states to undo."""
        return bool(self.undo_stack)

    def can_redo(self):
        """Check if there are states to redo."""
        return bool(self.redo_stack)

