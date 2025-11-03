# Range Selection Seek Behavior

## Feature Description

When the user interacts with timeline ranges, the playhead (red tracking indicator) now automatically seeks to provide intuitive navigation:

1. **Click on a range** → Playhead immediately jumps to the start of that range
2. **Click on timeline axis** → Playhead jumps to that position (even if a range is selected)

This provides immediate visual feedback and makes it easy to review specific ranges.

## User Experience

### Before This Feature

**Old behavior:**
1. User clicks on a range → range highlights
2. User presses Play → video seeks to range start, then plays
3. Playhead only moved when Play was pressed

**Problems:**
- No immediate feedback when clicking range
- Had to press Play to see where the range starts
- Couldn't preview range location without starting playback

### After This Feature

**New behavior - Clicking a Range:**
1. User clicks on a range → range highlights
2. **Playhead immediately jumps to range start** ✅
3. Video display shows the first frame of the range
4. User can see exactly where the range starts
5. Pressing Play starts from there (already positioned)

**New behavior - Clicking Timeline:**
1. User clicks on a range (playhead at range start)
2. User clicks on timeline axis → **playhead jumps to clicked position** ✅
3. Timeline click overrides range selection positioning
4. User can freely navigate while range is still selected

## Implementation

### Part 1: Seek on Range Selection (app_controller.py:747-752)

```python
if is_selected and self.selected_timeline_range_data:
    action_code = self.selected_timeline_range_data.get('action', 'Unknown')
    status = self.selected_timeline_range_data.get('status')
    can_edit = self.window.timeline_widget._editing_enabled if self.window and self.window.timeline_widget else False
    
    # Seek to range start when range is selected
    if selection_changed and self.playback_manager:
        range_start_frame = self.selected_timeline_range_data.get('start')
        if range_start_frame is not None:
            print(f"Controller: Range selected - seeking to range start (F{range_start_frame})")
            self.playback_manager.seek(range_start_frame)
```

**Key points:**
- Only seeks when `selection_changed` is True (prevents repeated seeks)
- Seeks immediately when range is selected (not waiting for Play button)
- Uses existing `playback_manager.seek()` method

### Part 2: Simplified Play Button (app_controller.py:338-339)

```python
# OLD CODE - seek logic in play button:
if self.current_selected_index != -1 and self.selected_timeline_range_data:
    range_start_frame = self.selected_timeline_range_data.get('start')
    current_frame = self.playback_manager.current_frame
    if range_start_frame is not None and current_frame != range_start_frame:
        print(f"Controller: Play pressed with range selected - seeking to range start...")
        self.playback_manager.seek(range_start_frame)
        QTimer.singleShot(100, self.playback_manager.play)
        return
self.playback_manager.play()

# NEW CODE - just play:
# Just play - seeking already happened when range was selected
self.playback_manager.play()
```

**Why this works:**
- Seeking happens at selection time (immediately)
- Play button doesn't need to check or seek
- Cleaner separation of concerns

### Part 3: Timeline Click Already Works (timeline_widget.py:291-301)

The timeline widget already had proper seek handling:

```python
# Check if clicking in SEEK AREA (axis) FIRST, before range selection
track_y_start = self._padding + self._axis_height
is_in_seek_area = (can_seek and
                   event.button() == Qt.LeftButton and
                   self._padding <= pos.x() <= self.width() - self._padding and
                   pos.y() < track_y_start)

if is_in_seek_area:
    # Clicking in axis area → SEEK (don't select ranges)
    self.seek_requested.emit(frame)
    return  # Exit early - don't process range selection
```

**How it works:**
- Timeline axis (top area) is checked **before** range selection logic
- Clicking axis emits `seek_requested` signal
- Signal is connected to `playback_manager.seek()` (line 246)
- Works regardless of whether a range is selected

## Signal Flow

### Flow 1: User Clicks Range

```
1. User clicks range in timeline
   ↓
2. timeline_widget detects click on range
   ↓
3. timeline_widget.range_selected.emit(True, range_data)
   ↓
4. app_controller._handle_timeline_range_selected(True, range_data)
   ↓
5. selection_changed = True (new range selected)
   ↓
6. playback_manager.seek(range_start_frame)
   ↓
7. qt_media_player.seek(range_start_frame)
   ↓
8. Video seeks, timeline updates, playhead moves ✅
```

### Flow 2: User Clicks Timeline Axis (after selecting range)

```
1. User clicks timeline axis (not on a range)
   ↓
2. timeline_widget.mousePressEvent checks seek area first
   ↓
3. is_in_seek_area = True (clicked on axis)
   ↓
4. timeline_widget.seek_requested.emit(frame)
   ↓
5. playback_manager.seek(frame) - directly connected
   ↓
6. Video seeks to clicked position ✅
   ↓
7. Range remains selected (but playhead moved)
```

## What Now Works ✅

**Immediate Visual Feedback:**
- Click range → playhead **instantly** jumps to range start
- Video display shows first frame of range
- Timeline indicator moves to correct position
- No need to press Play to see where range is

**Flexible Navigation:**
- Click range → playhead at range start
- Click timeline axis → playhead at clicked position
- Range stays selected even after timeline click
- Can review different parts while range is selected

**Intuitive Playback:**
- Click range → already positioned at start
- Press Play → starts immediately (no seek delay)
- Natural workflow for reviewing ranges

## Use Cases

### Use Case 1: Reviewing Action Ranges
1. Click on "Smile" range → playhead jumps to smile start, see first frame
2. Press Play → watch smile action from beginning
3. Natural, immediate feedback

### Use Case 2: Comparing Ranges
1. Click range A → see start of range A
2. Click range B → see start of range B
3. Quick comparison without playing

### Use Case 3: Editing Workflow
1. Click range to select it
2. See where it starts on the video
3. Use arrow keys or timeline to navigate and adjust
4. Range stays selected for editing

### Use Case 4: Mixed Navigation
1. Click range → playhead at range start
2. Click timeline axis → playhead at different position
3. Still can edit the selected range
4. Flexible navigation while keeping selection

## Files Modified

- `S2 Action Coder/app_controller.py` (lines 338-339, 747-752)
- `S2 Action Coder/timeline_widget.py` (no changes - already works correctly)

## Testing

1. Open a video with ranges
2. Click on a range
3. **Verify:** 
   - Playhead immediately jumps to range start
   - Video shows first frame of range
   - Timeline indicator at correct position

4. Click on timeline axis (not on a range)
5. **Verify:**
   - Playhead jumps to clicked position
   - Range remains selected (highlighted)
   - Can still edit/delete range

6. Press Play
7. **Verify:**
   - Video plays from current position
   - No seeking delay
   - Smooth playback start

## Technical Notes

**Why seek on selection instead of play?**
- Immediate visual feedback is more intuitive
- User sees where range starts right away
- Play button is simpler (just plays, doesn't seek)
- Better separation of concerns

**Why does timeline click work while range is selected?**
- Timeline axis click is checked **before** range selection logic
- This is intentional - allows free navigation
- Range selection and seek position are independent concepts
- Range = what you're editing, Position = where you're viewing

**Performance considerations:**
- Seeking on selection has same performance as seeking on play
- Actually faster overall (no extra check in play button)
- No additional delays or timers needed
