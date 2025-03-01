# action_tracker.py - Tracks which actions are active during which frames
class ActionTracker:
    def __init__(self):
        """Initialize the action tracker."""
        self.actions = {}  # Frame -> Action code mapping
        self.current_frame = 0
        self.active_action = None
    
    def set_frame(self, frame):
        """Set the current frame number."""
        self.current_frame = frame
    
    def start_action(self, action_code):
        """Start an action at the current frame."""
        self.active_action = action_code
        self.actions[self.current_frame] = action_code
    
    def continue_action(self):
        """Continue the active action to the current frame if one is active."""
        if self.active_action:
            self.actions[self.current_frame] = self.active_action
    
    def stop_action(self):
        """Stop the active action."""
        self.active_action = None
    
    def get_action_for_frame(self, frame):
        """Get the action for a specific frame."""
        return self.actions.get(frame, "")
    
    def get_all_actions(self):
        """Get the complete action mapping dictionary."""
        return self.actions
    
    def clear_actions(self):
        """Clear all tracked actions."""
        self.actions = {}
        self.active_action = None
    
    def save_actions(self, file_path):
        """Save actions to a file for later loading."""
        import json
        try:
            # Convert frame numbers from int to string for JSON
            actions_str_keys = {str(k): v for k, v in self.actions.items()}
            with open(file_path, 'w') as f:
                json.dump(actions_str_keys, f)
            return True
        except Exception as e:
            print(f"Error saving actions: {e}")
            return False
    
    def load_actions(self, file_path):
        """Load actions from a file."""
        import json
        try:
            with open(file_path, 'r') as f:
                actions_str_keys = json.load(f)
            # Convert frame numbers from string back to int
            self.actions = {int(k): v for k, v in actions_str_keys.items()}
            return True
        except Exception as e:
            print(f"Error loading actions: {e}")
            return False
