# Play from Range Start Feature

## Feature Description

When a timeline range is selected and the user presses the Play button, playback now automatically starts at the beginning of that selected range.

## User Experience

### Before This Feature
1. User clicks on a range in the timeline → range highlights
2. User presses Play → video plays from current frame position
3. User had to manually seek to the start of the range to play it from the beginning

### After This Feature
1. User clicks on a range in the timeline → range highlights
2. User presses Play → video automatically seeks to range start, then plays
3. Convenient for reviewing specific action ranges

## Implementation

**File:** `app_controller.py`  
**Method:** `_toggle_playback()` (lines 338-344)

```python
# If a range is selected, seek to its start before playing
if self.current_selected_index != -1 and self.selected_timeline_range_data:
    range_start_frame = self.selected_timeline_range_data.get('start')
    current_frame = self.playback_manager.current_frame
    if range_start_frame is not None and current_frame != range_start_frame:
        print(f"Controller: Play pressed with range selected - seeking to range start (F{range_start_frame})")
        self.playback_manager.seek(range_start_frame)

self.playback_manager.play()
```

## Behavior Details

- **Only seeks when needed:** If playback head is already at the range start, no seeking occurs
- **Automatic:** No user configuration required - works automatically when range is selected
- **Non-intrusive:** If no range is selected, play button works normally (plays from current position)
- **Works with all range types:** TBC, NM, confirm_needed, and regular action ranges

## Use Cases

1. **Reviewing action ranges:** Click a range, press Play → immediately see the action from start to finish
2. **Confirming classifications:** Select a range to review whether it's correctly coded
3. **Quality checking:** Quickly jump to and play specific ranges for verification
4. **Editing workflow:** After editing a range's boundaries, press Play to preview the result

## Files Modified

- `S2 Action Coder/app_controller.py` (lines 338-344)

## Testing

1. Open a processed video with timeline ranges
2. Click on any range in the timeline
3. Press the Play button
4. **Expected:** Video seeks to the start of the selected range and begins playing
5. **Verify:** Log shows "Play pressed with range selected - seeking to range start (F###)"
