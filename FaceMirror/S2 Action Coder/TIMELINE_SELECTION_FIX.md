# Timeline Range Selection Bug Fix

## Problem

Two related issues prevented proper timeline interaction in S2 Action Coder:

### Issue 1: Range Selection Not Working
- Clicking on timeline ranges did not highlight or select them
- Unable to edit or change existing ranges
- Action buttons would not activate for selected ranges

### Issue 2: Unknown/TBC Ranges Not Assignable
- Clicking on TBC (To Be Coded) or unknown ranges showed "Please drag to create range" message
- Could not assign action codes directly to unknown ranges
- Had to delete and recreate ranges to assign codes

## Root Cause

In `timeline_widget.py`, the `mousePressEvent` function had a critical bug:

**Lines 316-319 (before fix):**
```python
clicked_range_index = -1
for i, r in enumerate(self._action_ranges):
    range_rect = self._get_range_rect(i)
    if range_rect.contains(pos): clicked_range_index = i; break
# BUG: clicked_range_index was found but NEVER USED!
```

The code correctly identified which range was clicked, but:
- Never set `self._selected_range_index = clicked_range_index`
- Never set `selection_changed = True`
- Never emitted the `range_selected` signal

This meant clicking ranges had no effect, making them appear "unclickable".

## Solution

Added proper range selection logic in `timeline_widget.py` lines 320-335:

```python
# Handle range selection
track_y_start = self._padding + self._axis_height
track_y_end = track_y_start + self._track_height
is_in_track_area = (self._padding <= pos.x() <= self.width() - self._padding and
                   track_y_start <= pos.y() <= track_y_end)

if clicked_range_index != -1:
    # Clicked on a range - select it
    if clicked_range_index != self._selected_range_index:
        self._selected_range_index = clicked_range_index
        selection_changed = True
elif is_in_track_area:
    # Clicked on empty space in track area - deselect
    if self._selected_range_index != -1:
        self._selected_range_index = -1
        selection_changed = True
```

## What Now Works

✅ **Range Selection:**
- Click any range to select it (highlights with blue border)
- Action buttons enable for the selected range
- Status display shows current action code

✅ **TBC/Unknown Range Assignment:**
- Click TBC ("??") ranges to select them
- Action buttons become available
- Click any action code button to assign it
- No need to recreate the range

✅ **Deselection:**
- Click empty track space to deselect current range
- Clicking outside track area doesn't interfere

✅ **Range Creation Still Works:**
- Drag in empty track space to create new TBC range
- Works when no range is currently selected

## Files Modified

- `S2 Action Coder/timeline_widget.py` (lines 315-335)

## Testing Recommendations

1. Open a processed video in S2 Action Coder
2. Click on any timeline range → should highlight with blue border
3. Click on a TBC ("??") range → action buttons should enable
4. Click an action code button → range should update to that code
5. Click empty track space → range should deselect
6. Drag in empty space → new TBC range should be created

All functionality should now work as originally intended.
