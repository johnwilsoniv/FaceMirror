# Batch File Removal on Save Feature

## Feature Description

When a user completes and saves a video in batch processing mode, that video is now **automatically removed from the batch**. The total file count decreases, providing a clear indication of progress through the batch.

## User Experience

### Before This Feature

**Old behavior:**
```
Batch Navigation: File 25 of 59
[User saves and continues]
Batch Navigation: File 26 of 59  (total stays at 59)
```

**Problems:**
- Total count never decreased
- Hard to see actual remaining work
- Completed files stayed in the batch list
- Could accidentally re-process completed files

### After This Feature

**New behavior:**
```
Batch Navigation: File 25 of 59
[User saves and continues]
Batch Navigation: File 25 of 58  (total decreased to 58!)
```

**Benefits:**
- ✅ Total count decreases as work is completed
- ✅ Clear visual progress indicator
- ✅ Completed files removed from batch
- ✅ Can't accidentally re-process completed files
- ✅ See exactly how many files remain

## Use Cases

### Use Case 1: Long Batch Session
1. Start with 100 videos
2. Complete video #25 → 99 remain
3. Complete video #25 (next file slides into position) → 98 remain
4. Take a break, come back
5. See "File 25 of 98" - know exactly where you are

### Use Case 2: Multiple Users/Sessions
1. User A processes first 20 videos
2. User B starts later, sees "File 1 of 80"
3. Knows 20 were already completed
4. Clear division of work

### Use Case 3: Progress Tracking
1. Boss: "How many videos left?"
2. You: "File 45 of 123" (78 to go)
3. One hour later: "File 45 of 110" (65 to go)
4. Clear progress measurement

## Implementation

### Part 1: Remove Current File Method (batch_processor.py:245-264)

```python
def remove_current_file(self):
    """
    Remove the current file from the batch after completion.
    Adjusts the current_index to stay at the same position (which becomes the next file).

    Returns:
        True if file was removed, False if no current file or list is empty
    """
    if 0 <= self.current_index < len(self.file_sets):
        removed_file = self.file_sets.pop(self.current_index)
        print(f"BatchProcessor: Removed completed file '{removed_file.get('base_id', 'Unknown')}' from batch")
        print(f"BatchProcessor: Batch now has {len(self.file_sets)} files remaining")
        # Don't adjust index - the next file is now at the same index position
        # But if we removed the last file, move index back
        if self.current_index >= len(self.file_sets) and len(self.file_sets) > 0:
            self.current_index = len(self.file_sets) - 1
        elif len(self.file_sets) == 0:
            self.current_index = -1
        return True
    return False
```

**Key logic:**
- Removes current file from `file_sets` list
- Keeps `current_index` at same position
- Next file "slides" into the current position
- Handles edge case of removing last file

### Part 2: Call on Save Complete (app_controller.py:1023-1045)

```python
@pyqtSlot(bool)
def _handle_save_complete(self, success):
    if self.ui_manager: 
        self.ui_manager.show_progress_bar(False)

    # Track failed files
    if not success:
        # ... error handling ...
        return

    # Remove the completed file from the batch (decreases total count)
    if success:
        self.batch_processor.remove_current_file()
        # Update UI to show new batch count
        if self.ui_manager:
            current_idx = self.batch_processor.get_current_index()
            total_files = self.batch_processor.get_total_files()
            self.ui_manager.set_batch_navigation(True, current_idx, total_files)

    # Check if there are more files (after removal, current index points to what was the next file)
    total_remaining = self.batch_processor.get_total_files()
    if total_remaining > 0:
        # After removal, current_index points to the next file (which slid into current position)
        # So we load the current file, not the next file
        print(f"Controller: Save complete, loading next file (now at index {self.batch_processor.get_current_index()} of {total_remaining})...")
        gc.collect()
        next_file_set = self.batch_processor.get_current_file_set()
        QTimer.singleShot(100, lambda: self.load_batch_file_set(next_file_set))
    else:
        print("Controller: Save complete. Batch finished.")
        self._show_batch_completion_summary(incomplete=False)
```

**Flow:**
1. Save completes successfully
2. Remove current file from batch
3. Update UI with new count
4. Load next file (which is now at current index)
5. Or show completion summary if no files remain

## Technical Details

### Index Management

**Scenario: Removing file at index 5 from a list of 60:**

```
Before removal:
file_sets = [f0, f1, f2, f3, f4, f5, f6, f7, ...]  (60 total)
current_index = 5 (pointing to f5)

After removal:
file_sets = [f0, f1, f2, f3, f4, f6, f7, f8, ...]  (59 total)
current_index = 5 (now pointing to f6 - what was the next file!)
```

**Why keep index the same?**
- The "next" file slides into the current position
- Simplifies logic - just load current_file_set
- Natural progression through the list

**Edge cases handled:**
- Removing last file: index adjusted to len(file_sets) - 1
- Removing last remaining file: index set to -1
- Empty list: returns False, no crash

### Load Next File Logic

**Old approach (before batch removal):**
```python
if self.batch_processor.has_next_file():
    QTimer.singleShot(100, self.load_next_file)  # Calls load_next_file()
```

**New approach (with batch removal):**
```python
if total_remaining > 0:
    next_file_set = self.batch_processor.get_current_file_set()  # Get current position
    QTimer.singleShot(100, lambda: self.load_batch_file_set(next_file_set))
```

**Why the change?**
- After removal, `current_index` already points to next file
- Calling `load_next_file()` would skip a file
- Direct `get_current_file_set()` loads the correct file

## What Now Works ✅

**Progress Tracking:**
- Batch starts: "File 1 of 100"
- After 10 completions: "File 1 of 90"
- Clear remaining work indicator

**UI Updates:**
- Total count decreases immediately after save
- Current index stays consistent or adjusts appropriately
- Navigation buttons work correctly

**Error Handling:**
- Failed saves don't remove file from batch
- User can retry failed file
- Only successful saves trigger removal

**Completion:**
- Last file completed → batch count goes to 0
- Completion summary shows correctly
- No off-by-one errors

## Files Modified

- `S2 Action Coder/batch_processor.py` (lines 245-264)
- `S2 Action Coder/app_controller.py` (lines 1023-1045)

## Testing Scenarios

### Test 1: Normal Progression
1. Load batch of 10 files
2. Verify: "File 1 of 10"
3. Complete file 1 (save and continue)
4. **Verify:** "File 1 of 9" (not "File 2 of 10")
5. Complete file 1 (new file in position 1)
6. **Verify:** "File 1 of 8"

### Test 2: Last File
1. Complete all files until 1 remains
2. Verify: "File 1 of 1"
3. Complete last file
4. **Verify:** Completion summary shows
5. No crash or error

### Test 3: Failed Save
1. Start with "File 1 of 5"
2. Save fails (simulate by causing error)
3. **Verify:** Still shows "File 1 of 5" (not removed)
4. Can retry the same file

### Test 4: Skip Without Saving
1. "File 1 of 5"
2. Click "Next" (without saving)
3. **Verify:** Shows "File 2 of 5" (total unchanged)
4. Skipped file still in batch
5. Can go back to it later

## Future Enhancements

Possible improvements for later:

1. **Skip & Remove:** Option to remove file from batch without saving (mark as skipped)
2. **Undo Remove:** Ability to re-add a completed file if needed
3. **Persistent State:** Save batch state so removal persists across app restarts
4. **Progress Bar:** Visual progress bar showing X of Y completed
5. **Export Batch State:** Save list of completed vs. remaining files

## Design Decisions

**Why remove on save, not on skip?**
- Save indicates completion
- Skip might be temporary (user wants to come back)
- Clear distinction between "completed" and "deferred"

**Why not persist removals?**
- Keeps initial implementation simple
- App restart resets batch (can re-process if needed)
- Future enhancement if needed

**Why update UI immediately?**
- Instant feedback for user
- Clear progress indication
- Matches user's mental model of "completion"
