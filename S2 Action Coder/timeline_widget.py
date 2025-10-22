
from PyQt5.QtWidgets import QWidget, QOpenGLWidget, QApplication, QMenu, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QRect, QPoint, QSize, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFontMetrics, QFont, QCursor, QKeySequence # Added QKeySequence

import config
import math
import copy
import time  # For performance timing logging
from perf_logger import log_perf_warning, log_perf_info  # Centralized performance logging

class TimelineWidget(QOpenGLWidget):
    ranges_edited = pyqtSignal(list, int, str) # list: new ranges, int: dragged_index (-1 if create), str: drag_type
    delete_range_requested = pyqtSignal(object) # object: range_data dict to delete
    seek_requested = pyqtSignal(int) # int: frame_number
    range_selected = pyqtSignal(bool, object) # bool: is_selected, object: range_data dict or None

    def __init__(self, parent=None):
        # (Initialization unchanged)
        super().__init__(parent); self.setMinimumHeight(120); self.setFocusPolicy(Qt.StrongFocus); self.setMouseTracking(True)
        self._action_ranges = []; self._total_frames = 1; self._fps = 30.0; self._current_frame = 0
        self._previous_playhead_x = -1  # Track previous playhead position for partial updates
        self._padding = 10; self._track_height = 50
        self._axis_height = 30; self._playhead_color = QColor(Qt.red)
        self._background_color = QColor(config.UI_COLORS.get('section_bg', Qt.white))
        self._track_background_color = QColor(config.UI_COLORS.get('timeline_track_bg', Qt.white))
        self._axis_color = QColor(Qt.black); self._tick_color = QColor(Qt.gray); self._action_colors = self._generate_action_colors()
        self._selected_range_color = QColor(Qt.yellow)
        self._confirm_needed_border_color = QColor(config.UI_COLORS.get('timeline_confirm_needed_border', Qt.red))
        self._tbc_nm_text_color = QColor(config.UI_COLORS.get('timeline_tbc_nm_text', Qt.darkGray))
        self._edge_grab_margin = 5
        self._selected_range_index = -1; self._dragging_mode = None; self._drag_start_pos = None
        self._drag_start_frame = 0; self._drag_start_frame_offset = 0; self._temp_drag_rect = None
        self._hovered_range_index = -1; self._hovered_edge = None; self._original_ranges_during_drag = None
        self._temp_ranges_for_visual_feedback = None
        self._editing_enabled = False
        self._playing_range_index = -1  # NEW: Track which range is currently playing for visual feedback

        # === PERFORMANCE OPTIMIZATION: Cache font objects to avoid recreating per range ===
        self._label_font = QFont("Arial", 9)
        self._label_font_metrics = QFontMetrics(self._label_font)
        self._tick_font = QFont("Arial", 8)
        self._tick_font_metrics = QFontMetrics(self._tick_font)
        # === END OPTIMIZATION ===

    # (_generate_action_colors - unchanged)
    def _generate_action_colors(self): # (Unchanged)
        colors = {}; predefined_colors = [QColor("#1f77b4"), QColor("#ff7f0e"), QColor("#2ca02c"), QColor("#d62728"), QColor("#9467bd"), QColor("#8c564b"), QColor("#e377c2"), QColor("#7f7f7f"), QColor("#bcbd22"), QColor("#17becf"), QColor("#aec7e8"), QColor("#ffbb78"), QColor("#98df8a"), QColor("#ff9896"), QColor("#c5b0d5"), QColor("#c49c94")]; i = 0
        colors['BL'] = QColor(config.UI_COLORS.get('timeline_bl_color', '#dddddd'))
        colors['TBC'] = QColor(config.UI_COLORS.get('timeline_tbc_color', '#fff3cd'))
        colors['NM'] = QColor(config.UI_COLORS.get('timeline_nm_color', '#cfe2ff'))
        for code in sorted(config.ACTION_MAPPINGS.keys()):
            if code not in colors and code != 'SO_SE': colors[code] = predefined_colors[i % len(predefined_colors)]; i += 1
        return colors

    # --- MODIFIED update_ranges - Removed premature return ---
    @pyqtSlot(list)
    def update_ranges(self, action_ranges):
         print(f"DEBUG: TimelineWidget update_ranges called with {len(action_ranges)} ranges.") # DEBUG
         if self._dragging_mode:
             print("DEBUG: TimelineWidget update_ranges ignored (dragging).") # DEBUG
             return

         # --- REMOVED CHECK: Always update internal state and repaint ---
         # current_copy = copy.deepcopy(self._action_ranges)
         # new_copy = copy.deepcopy(action_ranges)
         # needs_update = (current_copy != new_copy)
         # if not needs_update:
         #     print("DEBUG: TimelineWidget update_ranges ignored (no change detected).") # DEBUG
         #     return
         # --- END REMOVED CHECK ---

         # Set internal ranges
         self._action_ranges = sorted(copy.deepcopy(action_ranges), key=lambda x: x.get('start', 0)) if action_ranges else []
         print(f"DEBUG: TimelineWidget internal _action_ranges set to {len(self._action_ranges)} ranges.") # DEBUG

         # Clear selection if the current index becomes invalid
         if not (0 <= self._selected_range_index < len(self._action_ranges)):
             if self._selected_range_index != -1:
                 print(f"DEBUG: Clearing invalid selection index ({self._selected_range_index}) during update_ranges.") # DEBUG
                 original_index = self._selected_range_index
                 self._selected_range_index = -1
                 # Don't re-emit range_selected(False) here if it was already cleared by controller
                 # self.range_selected.emit(False, None)

         # Request repaint
         print("DEBUG: TimelineWidget calling self.update() for repaint.") # DEBUG
         self.update()


    # Video and frame navigation methods (optimized for performance)
    @pyqtSlot(int, int, float)
    def set_video_properties(self, total_frames, width, fps):
         self._total_frames = max(1, total_frames); self._fps = fps if fps > 0 else 30.0
         self.update()
    @pyqtSlot(int)
    def set_current_frame(self, frame_number):
        new_frame = max(0, min(frame_number, self._total_frames - 1))
        if new_frame != self._current_frame:
            self._current_frame = new_frame
            # Partial update: only repaint playhead regions
            playhead_width = 5  # Playhead line width + margin
            # Clear old playhead position
            if self._previous_playhead_x >= 0:
                old_rect = QRect(self._previous_playhead_x - playhead_width, 0, playhead_width * 2, self.height())
                self.update(old_rect)
            # Draw new playhead position
            new_playhead_x = self.frame_to_x(new_frame)
            new_rect = QRect(new_playhead_x - playhead_width, 0, playhead_width * 2, self.height())
            self.update(new_rect)
            self._previous_playhead_x = new_playhead_x
    @pyqtSlot(bool)
    def set_editing_enabled(self, enabled): # (Unchanged)
        self._editing_enabled = enabled; print(f"TimelineWidget: Editing {'enabled' if enabled else 'disabled'}.")
        if not enabled: self._hovered_range_index = -1; self._hovered_edge = None; self.setCursor(Qt.ArrowCursor)
        current_data = None
        if self._selected_range_index != -1 and 0 <= self._selected_range_index < len(self._action_ranges): current_data = self._action_ranges[self._selected_range_index]
        self.range_selected.emit(self._selected_range_index != -1, current_data)

    # NEW: Set playing range for visual feedback
    def set_playing_range(self, range_index):
        """Set which range is currently playing (for visual feedback)"""
        if self._playing_range_index != range_index:
            self._playing_range_index = range_index
            self.update()  # Trigger repaint to show visual feedback

    def _get_view_width(self):
        return self.width() - 2 * self._padding

    def _get_pixels_per_frame(self):
        """Calculate pixels per frame to fit entire timeline in view"""
        view_width = self._get_view_width()
        if view_width <= 0: return 1
        if self._total_frames <= 0: return 1
        return max(0.01, view_width / self._total_frames)

    def frame_to_x(self, frame):
        """Convert frame number to x-coordinate"""
        pixels_per_frame = self._get_pixels_per_frame()
        return self._padding + int(frame * pixels_per_frame)

    def x_to_frame(self, x_pos):
        """Convert x-coordinate to frame number"""
        view_width = self._get_view_width()
        if view_width <= 0: return 0
        pixels_per_frame = self._get_pixels_per_frame()
        if pixels_per_frame <= 0: return 0
        adjusted_x = max(0, x_pos - self._padding)
        frame = int(adjusted_x / pixels_per_frame)
        return max(0, min(frame, self._total_frames - 1))
    def _get_range_rect(self, range_index, use_temp_ranges=False): # (Unchanged)
        ranges_to_use = self._temp_ranges_for_visual_feedback if use_temp_ranges and self._temp_ranges_for_visual_feedback is not None else self._action_ranges
        if range_index < 0 or range_index >= len(ranges_to_use): return QRect()
        r = ranges_to_use[range_index]
        start_frame = r.get('start'); end_frame = r.get('end')
        if start_frame is None or end_frame is None: return QRect()
        x_start = self.frame_to_x(start_frame); x_end = self.frame_to_x(end_frame + 1)
        width = max(1, x_end - x_start); track_y = self._padding + self._axis_height
        return QRect(x_start, track_y, width, self._track_height)
    def _draw_single_range(self, painter, range_index, ranges_to_draw): # (Unchanged)
        if range_index < 0 or range_index >= len(ranges_to_draw): return
        r = ranges_to_draw[range_index]; action_code = r.get('action'); status = r.get('status')
        if not action_code: return
        range_rect = self._get_range_rect(range_index, use_temp_ranges=(self._dragging_mode and self._temp_ranges_for_visual_feedback is not None))
        if not range_rect.isValid() or range_rect.width() <= 0: return
        view_width = self._get_view_width(); track_y = self._padding + self._axis_height
        visible_rect = range_rect.intersected(QRect(self._padding, track_y, view_width, self._track_height))
        if visible_rect.isEmpty(): return
        base_color = self._action_colors.get(action_code, QColor(Qt.darkGray)); pen_color = base_color.darker(130); brush_color = base_color; text_color = Qt.black
        display_text = action_code; pen_style = Qt.SolidLine; pen_width = 1
        if action_code in ["TBC", "NM"]: display_text = "??"; text_color = self._tbc_nm_text_color
        elif status == 'confirm_needed': display_text = f"{action_code}?"; pen_color = self._confirm_needed_border_color; pen_width = 2; brush_color = base_color.lighter(110); text_color = self._confirm_needed_border_color
        is_selected = (range_index == self._selected_range_index)
        is_playing = (range_index == self._playing_range_index)  # NEW: Check if this range is currently playing

        # Apply visual styling based on state
        if is_playing:  # NEW: Playing range gets green thick border
            pen_color = QColor(Qt.green); pen_width = 3; pen_style = Qt.SolidLine; brush_color = base_color.lighter(120)
        elif is_selected:  # Selected range gets yellow border
            pen_color = self._selected_range_color; pen_width = 2; pen_style = Qt.SolidLine; brush_color = base_color.lighter(110)

        painter.setPen(QPen(pen_color, pen_width, pen_style)); painter.setBrush(brush_color); painter.drawRect(visible_rect)

        # === OPTIMIZATION: Use cached font instead of creating new one ===
        painter.setFont(self._label_font)
        painter.setPen(text_color)
        text_rect = QRect(visible_rect.left() + 3, visible_rect.top(), visible_rect.width() - 6, visible_rect.height())
        elided_text = self._label_font_metrics.elidedText(display_text or "", Qt.ElideRight, text_rect.width())
        if elided_text:
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, elided_text)
        # === END OPTIMIZATION ===
    def paintEvent(self, event):
        # === PERFORMANCE LOGGING: Track paint event duration ===
        _start_time = time.time()

        painter = QPainter(self)
        # Disable antialiasing for performance (OpenGL handles smooth rendering)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), self._background_color)
        track_y = self._padding + self._axis_height; view_width = self._get_view_width()
        painter.fillRect(self._padding, track_y, view_width, self._track_height, self._track_background_color); painter.setPen(self._tick_color); painter.drawRect(self._padding, track_y, view_width, self._track_height)

        _t1 = time.time()
        ranges_to_draw = self._temp_ranges_for_visual_feedback if self._dragging_mode and self._temp_ranges_for_visual_feedback else self._action_ranges
        is_dragging_range = self._dragging_mode in ["start_edge", "end_edge", "move"] and self._selected_range_index != -1

        # === OPTIMIZATION: Viewport culling - only draw visible ranges ===
        # Calculate visible frame range
        first_visible_frame = self.x_to_frame(self._padding)
        last_visible_frame = self.x_to_frame(self.width() - self._padding)

        # Filter to only visible ranges
        visible_range_indices = []
        for i, r in enumerate(ranges_to_draw):
            start_f = r.get('start')
            end_f = r.get('end')
            if start_f is not None and end_f is not None:
                # Check if range overlaps visible viewport
                if end_f >= first_visible_frame and start_f <= last_visible_frame:
                    visible_range_indices.append(i)

        _range_count = len(visible_range_indices)  # Count only visible ranges

        # Draw visible ranges only
        if is_dragging_range:
            # Draw non-selected ranges first
            for i in visible_range_indices:
                if i != self._selected_range_index:
                    self._draw_single_range(painter, i, ranges_to_draw)
            # Draw selected range last (on top)
            if self._selected_range_index in visible_range_indices:
                self._draw_single_range(painter, self._selected_range_index, ranges_to_draw)
        else:
            for i in visible_range_indices:
                self._draw_single_range(painter, i, ranges_to_draw)
        # === END OPTIMIZATION ===
        _t2 = time.time()
        axis_y = self._padding + self._axis_height - 1
        painter.setPen(QPen(self._axis_color, 1)); painter.drawLine(self._padding, axis_y, self.width() - self._padding, axis_y)
        pixels_per_frame = self._get_pixels_per_frame(); major_tick_interval_frames = 1
        if self._fps > 0 and pixels_per_frame > 0:
            pixels_per_sec = pixels_per_frame * self._fps
            if pixels_per_sec < 5: major_tick_interval_frames = int(self._fps * 10)
            elif pixels_per_sec < 20: major_tick_interval_frames = int(self._fps * 5)
            elif pixels_per_sec < 100: major_tick_interval_frames = int(self._fps * 1)
            elif pixels_per_frame < 3: major_tick_interval_frames = max(1, int(self._fps / 2))
            else: major_tick_interval_frames = 1
        major_tick_interval_frames = max(1, major_tick_interval_frames)
        first_visible_frame = self.x_to_frame(self._padding); last_visible_frame = self.x_to_frame(self.width() - self._padding)
        start_tick_frame = math.ceil(first_visible_frame / major_tick_interval_frames) * major_tick_interval_frames if major_tick_interval_frames > 0 else first_visible_frame

        # === OPTIMIZATION: Use cached tick font instead of creating new one ===
        painter.setFont(self._tick_font)
        last_label_end_x = -1
        # === END OPTIMIZATION ===
        if major_tick_interval_frames > 0:
            for frame in range(start_tick_frame, last_visible_frame + 1, major_tick_interval_frames):
                 x = self.frame_to_x(frame)
                 if x < self._padding or x > self.width() - self._padding: continue
                 painter.setPen(self._tick_color); painter.drawLine(x, axis_y - 5, x, axis_y + 5)
                 if self._fps > 0:
                     time_sec = frame / self._fps; label = f"{time_sec:.1f}s"
                     # === OPTIMIZATION: Use cached font metrics ===
                     label_width = self._tick_font_metrics.horizontalAdvance(label)
                     label_x = x - label_width // 2
                     if label_x > last_label_end_x + 5:
                         label_rect = QRect(label_x, self._padding, label_width, self._tick_font_metrics.height())
                         painter.setPen(self._axis_color)
                         painter.drawText(label_rect, Qt.AlignCenter, label)
                         last_label_end_x = label_x + label_width
                     # === END OPTIMIZATION ===
        playhead_x = self.frame_to_x(self._current_frame)
        if self._padding <= playhead_x <= self.width() - self._padding: painter.setPen(QPen(self._playhead_color, 2)); painter.drawLine(playhead_x, self._padding, playhead_x, self.height() - self._padding)
        if self._dragging_mode == "create" and self._temp_drag_rect: painter.setPen(QPen(Qt.blue, 1, Qt.DashLine)); painter.setBrush(QColor(0, 0, 255, 30)); painter.drawRect(self._temp_drag_rect)

        # === PERFORMANCE LOGGING: Report if paint event was slow ===
        _total_time = time.time() - _start_time
        if _total_time > 0.016:  # > 16ms (60 FPS threshold)
            log_perf_warning(f"Timeline paintEvent took {_total_time*1000:.1f}ms "
                  f"(RangeDrawing:{(_t2-_t1)*1000:.1f}ms for {_range_count} ranges)")
        # === END PERFORMANCE LOGGING ===
    def mousePressEvent(self, event):
        """
        Handle mouse press events.
        FIXED: Check seek area FIRST to prevent range selection from blocking seeks.
        """
        pos = event.pos(); frame = self.x_to_frame(pos.x()); can_edit = self._editing_enabled; can_seek = True
        self._drag_start_pos = pos; self._drag_start_frame = frame; self._original_ranges_during_drag = None; self._temp_ranges_for_visual_feedback = None
        self._dragging_mode = None; self._drag_start_frame_offset = 0; drag_initiated = False; selection_changed = False

        # === FIX: Check if clicking in SEEK AREA (axis) FIRST, before range selection ===
        track_y_start = self._padding + self._axis_height
        is_in_seek_area = (can_seek and
                           event.button() == Qt.LeftButton and
                           self._padding <= pos.x() <= self.width() - self._padding and
                           pos.y() < track_y_start)

        if is_in_seek_area:
            # Clicking in axis area → SEEK (don't select ranges)
            self.seek_requested.emit(frame)
            return  # Exit early - don't process range selection
        # === END FIX ===

        if can_edit and event.button() == Qt.LeftButton and self._selected_range_index != -1 and self._selected_range_index < len(self._action_ranges):
            selected_range_rect = self._get_range_rect(self._selected_range_index)
            if selected_range_rect.isValid() and selected_range_rect.contains(pos):
                r = self._action_ranges[self._selected_range_index];
                self._original_ranges_during_drag = copy.deepcopy(self._action_ranges);
                self._temp_ranges_for_visual_feedback = copy.deepcopy(self._action_ranges)
                start_edge_rect = QRect(selected_range_rect.left(), selected_range_rect.top(), self._edge_grab_margin, selected_range_rect.height())
                end_edge_rect = QRect(selected_range_rect.right() - self._edge_grab_margin, selected_range_rect.top(), self._edge_grab_margin, selected_range_rect.height())
                if start_edge_rect.contains(pos): self._dragging_mode = "start_edge"; self.setCursor(Qt.SizeHorCursor); drag_initiated = True;
                elif end_edge_rect.contains(pos): self._dragging_mode = "end_edge"; self.setCursor(Qt.SizeHorCursor); drag_initiated = True;
                else: self._dragging_mode = "move"; self._drag_start_frame_offset = self._drag_start_frame - r['start']; self.setCursor(Qt.SizeAllCursor); drag_initiated = True;
        if not drag_initiated and event.button() == Qt.LeftButton:
            clicked_range_index = -1
            for i, r in enumerate(self._action_ranges):
                range_rect = self._get_range_rect(i)
                if range_rect.contains(pos): clicked_range_index = i; break
        if not drag_initiated and not selection_changed and event.button() == Qt.LeftButton:
             track_y_start = self._padding + self._axis_height; track_y_end = track_y_start + self._track_height
             if can_edit and self._selected_range_index == -1 and self._padding <= pos.x() <= self.width() - self._padding and track_y_start <= pos.y() <= track_y_end:
                 self._dragging_mode = "create"; self._temp_drag_rect = QRect(pos, QSize(0, 0)); drag_initiated = True;
                 self._original_ranges_during_drag = copy.deepcopy(self._action_ranges); self._temp_ranges_for_visual_feedback = copy.deepcopy(self._action_ranges)
             # REMOVED: Old seek handling (now done at top of function to prevent range selection interference)
        if selection_changed:
             self.update();
             selected_data = None
             if 0 <= self._selected_range_index < len(self._action_ranges): selected_data = self._action_ranges[self._selected_range_index]
             self.range_selected.emit(self._selected_range_index != -1, selected_data)
        elif self._dragging_mode == "create": self.update()
    def mouseMoveEvent(self, event): # (Unchanged)
        pos = event.pos(); can_edit = self._editing_enabled
        if not self._dragging_mode:
            new_hovered_range_index = -1; new_hovered_edge = None; new_cursor = Qt.ArrowCursor
            for i, r in enumerate(self._action_ranges):
                 range_rect = self._get_range_rect(i)
                 if range_rect.isValid() and range_rect.contains(pos):
                      new_hovered_range_index = i
                      if can_edit:
                           start_edge_rect = QRect(range_rect.left(), range_rect.top(), self._edge_grab_margin, range_rect.height())
                           end_edge_rect = QRect(range_rect.right() - self._edge_grab_margin, range_rect.top(), self._edge_grab_margin, range_rect.height())
                           if start_edge_rect.contains(pos): new_cursor = Qt.SizeHorCursor; new_hovered_edge = "start"
                           elif end_edge_rect.contains(pos): new_cursor = Qt.SizeHorCursor; new_hovered_edge = "end"
                           else: new_cursor = Qt.SizeAllCursor
                      else: new_cursor = Qt.ArrowCursor; new_hovered_edge = None
                      break
            track_y_start = self._padding + self._axis_height
            if new_hovered_range_index == -1 and pos.y() < track_y_start and self._padding <= pos.x() <= self.width() - self._padding: new_cursor = Qt.PointingHandCursor
            if self.cursor().shape() != new_cursor: self.setCursor(new_cursor)
            self._hovered_range_index = new_hovered_range_index; self._hovered_edge = new_hovered_edge
        if can_edit and event.buttons() & Qt.LeftButton and self._dragging_mode:
            current_frame = self.x_to_frame(pos.x()); modified_visuals = False
            if self._dragging_mode in ["start_edge", "end_edge", "move"]:
                if self._selected_range_index != -1 and self._temp_ranges_for_visual_feedback is not None and self._selected_range_index < len(self._temp_ranges_for_visual_feedback):
                    temp_ranges = self._temp_ranges_for_visual_feedback; range_to_edit = temp_ranges[self._selected_range_index];
                    if self._original_ranges_during_drag and self._selected_range_index < len(self._original_ranges_during_drag): original_range_data = self._original_ranges_during_drag[self._selected_range_index]
                    else: original_range_data = range_to_edit
                    drag_original_start = original_range_data['start']; drag_original_end = original_range_data['end']; drag_original_duration = drag_original_end - drag_original_start
                    proposed_start = range_to_edit['start']; proposed_end = range_to_edit['end']
                    if self._dragging_mode == "move": proposed_start = current_frame - self._drag_start_frame_offset; proposed_end = proposed_start + drag_original_duration
                    elif self._dragging_mode == "start_edge": proposed_start = current_frame; proposed_end = drag_original_end
                    elif self._dragging_mode == "end_edge": proposed_end = current_frame; proposed_start = drag_original_start
                    new_start = max(0, proposed_start); new_end = min(self._total_frames - 1, proposed_end)
                    if new_start > new_end:
                        if self._dragging_mode == "start_edge": new_start = new_end
                        elif self._dragging_mode == "end_edge": new_end = new_start
                        elif self._dragging_mode == "move": new_start = range_to_edit['start']; new_end = range_to_edit['end']
                    if range_to_edit['start'] != new_start or range_to_edit['end'] != new_end: modified_visuals = True; range_to_edit['start'] = new_start; range_to_edit['end'] = new_end
                    if modified_visuals: self.update()
            elif self._dragging_mode == "create":
                 start_x = self.frame_to_x(self._drag_start_frame); current_x = max(self._padding, min(pos.x(), self.width() - self._padding))
                 track_y = self._padding + self._axis_height; self._temp_drag_rect = QRect(QPoint(min(start_x, current_x), track_y), QPoint(max(start_x, current_x), track_y + self._track_height)).normalized()
                 self.update()
    def mouseReleaseEvent(self, event): # (Unchanged)
        if event.button() == Qt.LeftButton and self._dragging_mode:
            print(f"Timeline STUCK_GUI_DEBUG: mouseReleaseEvent - START - Drag mode: {self._dragging_mode}, Selected index: {self._selected_range_index}")
            can_edit = self._editing_enabled; pos = event.pos(); end_frame_at_cursor = self.x_to_frame(pos.x())
            local_drag_mode = self._dragging_mode; local_selected_index = self._selected_range_index
            original_ranges = self._original_ranges_during_drag; temp_ranges = self._temp_ranges_for_visual_feedback
            drag_start_frame_local = self._drag_start_frame
            final_ranges_state = None; newly_created_range = None; newly_created_index = -1
            drag_occurred = (self._drag_start_pos != pos)
            if can_edit and local_drag_mode == "create" and drag_occurred:
                frame1 = min(drag_start_frame_local, end_frame_at_cursor); frame2 = max(drag_start_frame_local, end_frame_at_cursor)
                if frame2 > frame1:
                    new_range = {'action': 'TBC', 'start': frame1, 'end': frame2, 'status': None}
                    if temp_ranges is not None:
                        temp_ranges.append(new_range); temp_ranges.sort(key=lambda x: x.get('start', 0)); final_ranges_state = temp_ranges
                        try: newly_created_index = final_ranges_state.index(new_range)
                        except ValueError: print("Timeline ERROR: Could not find newly created TBC range after sorting."); newly_created_index = -1
                        newly_created_range = new_range; local_selected_index = newly_created_index
                    else: print("Timeline ERROR: temp_ranges was None during create release.")
            elif can_edit and local_drag_mode in ["start_edge", "end_edge", "move"] and drag_occurred:
                 if local_selected_index != -1 and temp_ranges is not None:
                     final_ranges_state = temp_ranges
            print(f"Timeline STUCK_GUI_DEBUG: mouseReleaseEvent - BEFORE CLEANUP - Drag mode: {local_drag_mode}, Final Index: {local_selected_index}, Changed State: {final_ranges_state is not None}")
            self._dragging_mode = None; self._original_ranges_during_drag = None; self._temp_ranges_for_visual_feedback = None; self._temp_drag_rect = None
            self._selected_range_index = local_selected_index
            self.setCursor(Qt.ArrowCursor); self.update()
            if final_ranges_state is not None:
                changed = False
                try:
                    if copy.deepcopy(final_ranges_state) != copy.deepcopy(original_ranges): changed = True
                except Exception as e: changed = True; print(f"Error comparing ranges on release: {e}")
                if changed:
                    self.ranges_edited.emit(copy.deepcopy(final_ranges_state), local_selected_index, local_drag_mode)
                if local_drag_mode == "create" and newly_created_range is not None:
                    self.range_selected.emit(True, newly_created_range) # Signal that the new range is selected
            print(f"Timeline STUCK_GUI_DEBUG: mouseReleaseEvent - END - Drag mode was: {local_drag_mode}")
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            # Navigate to previous frame
            new_frame = max(0, self._current_frame - 1)
            if new_frame != self._current_frame:
                self.seek_requested.emit(new_frame)
                event.accept()
            return
        elif key == Qt.Key_Right:
            # Navigate to next frame
            new_frame = min(self._total_frames - 1, self._current_frame + 1)
            if new_frame != self._current_frame:
                self.seek_requested.emit(new_frame)
                event.accept()
            return
        elif (key == Qt.Key_Delete or key == Qt.Key_Backspace) and self._editing_enabled:
            if self._selected_range_index != -1 and self._selected_range_index < len(self._action_ranges):
                 range_to_delete = self._action_ranges[self._selected_range_index]
                 print(f"TimelineWidget: Delete key pressed for range: {range_to_delete}")
                 self.delete_range_requested.emit(range_to_delete); event.accept()
            return
        elif event.matches(QKeySequence.Undo):
             print("TimelineWidget: Undo shortcut detected (forwarding to parent/controller)")
             event.ignore() # Let the main window handle it
             return
        elif event.matches(QKeySequence.Redo):
            print("TimelineWidget: Redo shortcut detected (forwarding to parent/controller)")
            event.ignore() # Let the main window handle it
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        hover_index = -1; pos = event.pos()
        for i, r in enumerate(self._action_ranges):
             range_rect = self._get_range_rect(i)
             if range_rect.contains(pos): hover_index = i; break
        if hover_index != -1 and self._editing_enabled:
            contextMenu = QMenu(self); deleteAction = contextMenu.addAction("Delete Range"); action = contextMenu.exec_(self.mapToGlobal(event.pos()))
            if action == deleteAction and hover_index < len(self._action_ranges):
                range_to_delete = self._action_ranges[hover_index]
                print(f"TimelineWidget: Context menu delete requested for range: {range_to_delete}")
                self.delete_range_requested.emit(range_to_delete)
    @pyqtSlot(int)
    def set_selected_range_by_index(self, index): # (Unchanged)
        new_index = -1; valid_index = False
        if 0 <= index < len(self._action_ranges): new_index = index; valid_index = True
        else: new_index = -1
        if self._selected_range_index != new_index:
            print(f"TimelineWidget: Setting selected index externally to {new_index}")
            self._selected_range_index = new_index
            self.update()

# --- END OF timeline_widget.py ---