# Timeline Tracking Indicator Fix

## Problem

When clicking on a timeline range and pressing Play:
- Video would correctly seek to the range start and begin playing
- BUT the red tracking indicator (playhead) on the timeline would not update
- The red line would stay at the old position instead of following playback
- This made it appear as if playback wasn't working properly

## Root Cause

**Timing issue between seek and play operations:**

1. User clicks range, then presses Play
2. Code calls: `playback_manager.seek(range_start_frame)`
3. Code immediately calls: `playback_manager.play()`
4. **Problem:** Seek operation has a 50ms delay before emitting frame update signal
5. Play starts before seek completes → timeline doesn't update to new position

### The Seek Delay

In `qt_media_player.py:327`, the seek operation uses a timer:
```python
def seek(self, frame):
    # ... setup ...
    self.media_player.setPosition(position_ms)
    
    # Force update after a short delay to allow position to settle
    QTimer.singleShot(50, lambda f=target_frame: self._force_update_frame_after_seek(f, was_playing))
```

This 50ms delay allows the media player's position to settle before emitting the frame update signal that updates the timeline widget.

### The Race Condition

**Old code flow:**
```
T=0ms:   seek(frame_100) called
T=0ms:   play() called immediately  
T=50ms:  seek completes, emits frame update (but play is already active)
Result:  Timeline shows old position, video plays from new position
```

**Fixed code flow:**
```
T=0ms:   seek(frame_100) called
T=50ms:  seek completes, emits frame update → timeline updates ✅
T=100ms: play() called via QTimer
Result:  Timeline shows correct position, video plays from correct position
```

## Solution

Added a 100ms delay before calling `play()` when seeking to a range start position.

**File:** `app_controller.py`  
**Method:** `_toggle_playback()` (lines 343-347)

```python
# If a range is selected, seek to its start before playing
if self.current_selected_index != -1 and self.selected_timeline_range_data:
    range_start_frame = self.selected_timeline_range_data.get('start')
    current_frame = self.playback_manager.current_frame
    if range_start_frame is not None and current_frame != range_start_frame:
        print(f"Controller: Play pressed with range selected - seeking to range start (F{range_start_frame})")
        self.playback_manager.seek(range_start_frame)
        # Delay play slightly to allow seek to complete and update timeline
        QTimer.singleShot(100, self.playback_manager.play)
        return  # Exit early, play will happen after delay

self.playback_manager.play()
```

## Why 100ms?

- Seek operation takes ~50ms to complete and emit frame update
- Added buffer of 50ms for signal propagation and UI updates
- Total 100ms is imperceptible to users (< 3 frames at 30fps)
- Ensures timeline indicator is always in sync with playback

## Signal Flow

The complete signal chain:
```
1. seek(frame_number) called
   ↓
2. [50ms delay] qt_media_player._force_update_frame_after_seek()
   ↓
3. qt_media_player.frameChanged.emit(frame, qimage, action)
   ↓
4. playback_manager._on_player_frame_changed(frame, qimage, action)
   ↓
5. playback_manager.frame_update_needed.emit(frame, qimage)
   ↓
6. timeline_widget.set_current_frame(frame)
   ↓
7. Timeline red indicator updates to new position ✅
   ↓
8. [100ms total] play() called
```

## What Now Works ✅

**Timeline Tracking:**
- Click range → Press Play → Red indicator immediately jumps to range start
- Red indicator follows playback smoothly during playback
- Timeline always shows accurate playback position

**User Experience:**
- No visual glitches or indicator lag
- Playback feels smooth and responsive
- Clear visual feedback of current position

## Files Modified

- `S2 Action Coder/app_controller.py` (lines 343-347)

## Testing

1. Open a video with timeline ranges
2. Click on any range (it highlights with blue border)
3. Press the Play button
4. **Verify:**
   - Red tracking indicator immediately jumps to range start
   - Red indicator smoothly follows playback progress
   - No lag or stuck indicator position

The 100ms delay is optimized for the seek operation's timing characteristics while remaining imperceptible to users.
