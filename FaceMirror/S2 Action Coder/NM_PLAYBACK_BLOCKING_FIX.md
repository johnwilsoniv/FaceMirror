# Near Miss (NM) Range Playback Blocking Bug Fix

## Problem

When clicking on existing Near Miss (NM) or confirm_needed ranges in the timeline:
- Playback would become blocked
- Play/pause button wouldn't work
- User couldn't navigate the video
- Debug log showed: `Play ignored - Waiting flags: Confirm=False, NM=True`

## Root Cause

In `app_controller.py`, the `_handle_timeline_range_selected` function treated **manual selection** of NM/confirm_needed ranges the same as **automatic prompts** during playback.

### Two Different Scenarios

**Scenario 1: Automatic Prompt (CORRECT behavior)**
- During video playback, Whisper processing detects a near miss
- `_trigger_near_miss_prompt()` is called automatically
- Sets `waiting_for_near_miss_assignment = True`
- Pauses playback and prompts user for confirmation
- This SHOULD block playback until user responds

**Scenario 2: Manual Selection (BROKEN behavior)**
- User clicks on an existing NM range to review/edit it
- Old code set `waiting_for_near_miss_assignment = True`
- Blocked playback incorrectly
- User just wanted to edit the range, not be prompted

## Solution

Modified `app_controller.py` to distinguish between automatic prompts and manual selection:

### Change 1: Always Clear Prompt Flags on Manual Selection (lines 717-724)
```python
# OLD CODE:
is_new_selection_a_prompt = False
if is_selected and self.selected_timeline_range_data:
    if self.selected_timeline_range_data.get('status') == 'confirm_needed' or 
       self.selected_timeline_range_data.get('action') == 'NM':
        is_new_selection_a_prompt = True
if not is_new_selection_a_prompt:
    # Clear flags

# NEW CODE:
# Manual selection of ranges (even NM or confirm_needed) should NOT be treated as prompts
# Prompts are only set via _trigger_next_confirmation or _trigger_near_miss_prompt during playback
# When manually selecting, always clear prompt state
if True:  # Always clear prompt state on manual selection
    if self.waiting_for_confirmation or self.waiting_for_near_miss_assignment:
        print("Controller STUCK_GUI_DEBUG: Clearing prompt flags...")
    self.waiting_for_confirmation = False
    self.waiting_for_near_miss_assignment = False
    self.current_confirmation_info = None
    self.current_near_miss_info = None
```

### Change 2: Don't Set Waiting Flags for Manual NM Selection (lines 741-743)
```python
# OLD CODE:
elif action_code == 'NM':
    print(f"Controller: Selected range is Near Miss (NM)")
    self.waiting_for_near_miss_assignment = True  # <-- WRONG!
    self.current_near_miss_info = {...}
    show_discard_nm = True
    ui_context = 'near_miss'
    enable_actions = True
    self._play_audio_segment(start_time, end_time)

# NEW CODE:
elif action_code == 'NM':
    # User manually selected an existing NM range - just enable editing, don't block playback
    print(f"Controller: Selected range is Near Miss (NM) - enabling edit mode")
    # DO NOT set waiting_for_near_miss_assignment = True here!
    # That flag is only for automatic prompts during playback (_trigger_near_miss_prompt)
    # Manual selection should allow normal playback/editing
    ui_context = 'selected'  # Treat like a normal selected range
    enable_actions = True
```

### Change 3: Same Fix for confirm_needed Ranges (lines 733-740)
```python
# OLD CODE:
if status == 'confirm_needed':
    print(f"Controller: Selected range needs confirmation: {action_code}")
    self.waiting_for_confirmation = True  # <-- WRONG!
    self.current_confirmation_info = self.selected_timeline_range_data
    show_discard_confirm = True
    ui_context = 'confirm'
    enable_actions = True
    self._play_audio_segment(start_time, end_time)

# NEW CODE:
if status == 'confirm_needed':
    # User manually selected an existing confirm_needed range - just enable editing
    print(f"Controller: Selected range with confirm_needed status: {action_code} - enabling edit mode")
    # DO NOT set waiting_for_confirmation = True here!
    # That flag is only for automatic prompts during playback (_trigger_next_confirmation)
    ui_context = 'selected'  # Treat like a normal selected range
    enable_actions = True
```

## What Now Works ✅

**Manual Selection of NM/confirm_needed Ranges:**
- Click on NM or confirm_needed ranges to select them
- Action buttons enable normally
- Can assign new action codes
- Can delete ranges
- Playback works normally (play/pause/seek)
- No blocking behavior

**Automatic Prompts Still Work:**
- During video playback, near miss detection still triggers prompts
- Confirmation prompts still work during playback
- These correctly pause and wait for user response

## Files Modified

- `S2 Action Coder/app_controller.py` (lines 717-747)

## Testing

1. Process a video with Whisper that creates NM ranges
2. After processing completes, click on an NM range
3. Verify:
   - Range highlights (blue border)
   - Action buttons enable
   - Play/pause button works
   - Can assign action codes to the NM range
   - No "Play ignored - Waiting flags" messages

The key insight: **Automatic prompts during playback** ≠ **Manual selection for editing**
