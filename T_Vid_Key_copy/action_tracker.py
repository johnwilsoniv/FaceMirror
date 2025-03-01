# action_tracker.py - Modified to use ranges rather than individual frames
class ActionTracker:
    def __init__(self):
        """Initialize the action tracker."""
        self.action_ranges = []  # List of dictionaries with 'action', 'start', 'end'
        self.current_frame = 0
        self.active_action = None
    
    def set_frame(self, frame):
        """Set the current frame number."""
        self.current_frame = frame
    
    def start_action(self, action_code):
        """Start an action at the current frame."""
        self.active_action = action_code
        # Add a new range with the current frame as start point
        self.action_ranges.append({
            'action': action_code,
            'start': self.current_frame,
            'end': None  # Will be set when action stops
        })
    
    def continue_action(self):
        """Continue the active action for the current frame."""
        # Nothing to do for range-based tracking, as ranges implicitly cover all frames
        pass
    
    def stop_action(self):
        """Stop the active action at the current frame."""
        if self.active_action and self.action_ranges:
            # Find the most recent range for this action and set its end point
            for i in range(len(self.action_ranges)-1, -1, -1):
                if self.action_ranges[i]['action'] == self.active_action and self.action_ranges[i]['end'] is None:
                    self.action_ranges[i]['end'] = self.current_frame
                    break
        self.active_action = None
    
    def get_action_for_frame(self, frame):
        """Get the action for a specific frame."""
        # Check all ranges to find the one that contains this frame
        for range_data in self.action_ranges:
            if range_data['start'] <= frame and (range_data['end'] is None or frame <= range_data['end']):
                return range_data['action']
        return ""
    
    def get_all_actions(self):
        """
        Get all actions for all frames.
        Returns a dictionary mapping frame numbers to action codes.
        """
        # Convert ranges to frame-by-frame mapping for compatibility with existing code
        frame_actions = {}
        for range_data in self.action_ranges:
            start = range_data['start']
            # If end is None (action is still active), use current frame as end
            end = range_data['end'] if range_data['end'] is not None else self.current_frame
            action = range_data['action']
            
            # Apply action to all frames in range
            for frame in range(start, end + 1):  # +1 to include end frame
                frame_actions[frame] = action
        
        return frame_actions
    
    def clear_actions(self):
        """Clear all tracked actions."""
        self.action_ranges = []
        self.active_action = None
    
    def save_actions(self, file_path):
        """Save actions to a file for later loading."""
        import json
        try:
            with open(file_path, 'w') as f:
                json.dump(self.action_ranges, f)
            return True
        except Exception as e:
            print(f"Error saving actions: {e}")
            return False
    
    def load_actions(self, file_path):
        """Load actions from a file."""
        import json
        try:
            with open(file_path, 'r') as f:
                self.action_ranges = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading actions: {e}")
            return False
            
    def validate_ranges(self):
        """
        Validate and clean up action ranges to ensure they don't overlap
        and have proper start/end values.
        """
        if not self.action_ranges:
            return
            
        # Sort ranges by start frame
        self.action_ranges.sort(key=lambda x: x['start'])
        
        # Set any None end values to current_frame
        for range_data in self.action_ranges:
            if range_data['end'] is None:
                range_data['end'] = self.current_frame
        
        # Validate that start <= end for all ranges
        for i, range_data in enumerate(self.action_ranges):
            if range_data['start'] > range_data['end']:
                # Swap start and end if they're in the wrong order
                self.action_ranges[i]['start'], self.action_ranges[i]['end'] = self.action_ranges[i]['end'], self.action_ranges[i]['start']
        
        # Check for and fix overlapping ranges (prioritize later ranges)
        for i in range(len(self.action_ranges) - 1):
            if self.action_ranges[i]['end'] >= self.action_ranges[i+1]['start']:
                # Trim the earlier range to end before the next one starts
                self.action_ranges[i]['end'] = self.action_ranges[i+1]['start'] - 1
                
                # If this creates an invalid range (start > end), remove it
                if self.action_ranges[i]['start'] > self.action_ranges[i]['end']:
                    self.action_ranges[i] = None
        
        # Remove any None ranges
        self.action_ranges = [r for r in self.action_ranges if r is not None]
        
        print(f"Validated {len(self.action_ranges)} action ranges")