# --- START OF FILE qt_media_player.py ---

# qt_media_player.py - Enhanced video player with reliable audio and video playback
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QUrl, Qt, QTimer
from PyQt5.QtGui import QImage
import cv2
import os
import time # For potential debugging delays

class QTMediaPlayer(QObject):
    """Enhanced media player with reliable audio/video playback."""
    frameChanged = pyqtSignal(int, QImage, str) # frame_num, qimage, action_code
    videoFinished = pyqtSignal()
    audioExtractionError = pyqtSignal(str)

    def __init__(self):
        super().__init__(); self.video_path = None; self.media_player = QMediaPlayer(); self.total_frames = 0
        self._current_frame_internal = 0; self.fps = 0; self.width = 0; self.height = 0; self.current_action = ""
        self.capture = None; self.container_widget = QWidget(); self.container_widget.setStyleSheet("background-color: black;")
        self.container_widget.setMinimumSize(640, 480); self.layout = QVBoxLayout(self.container_widget)
        self.layout.setContentsMargins(0, 0, 0, 0); self.layout.setSpacing(0); self.video_widget = QVideoWidget()
        self.layout.addWidget(self.video_widget);
        if self.media_player: self.media_player.setVideoOutput(self.video_widget)
        self.frame_timer = QTimer(); self.frame_timer.setTimerType(Qt.PreciseTimer); self.frame_timer.timeout.connect(self._on_frame_timer)
        if self.media_player:
            self.media_player.stateChanged.connect(self._on_state_changed); self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
            # We rely less on positionChanged now, using timer + getPosition instead
            # self.media_player.positionChanged.connect(self._on_position_changed)
        self.is_playing = False; self.last_emitted_frame = -1; self.frame_cache = {}; self.max_cache_size = 50; self.cache_hit_count = 0; self.cache_miss_count = 0
        # --- Add a flag to prevent recursive updates during seek ---
        self._is_seeking = False

    @property
    def current_frame(self):
        # Return the last frame we successfully emitted, otherwise the internal estimate
        return self.last_emitted_frame if self.last_emitted_frame >= 0 else self._current_frame_internal

    def set_video_path(self, video_path):
        print(f"Player: Loading video: {video_path}")
        if self.media_player: self.media_player.stop() # Stop existing playback/timer
        if not os.path.exists(video_path): print(f"Player Error: Video file does not exist at {video_path}"); return False
        self.video_path = video_path; self.frame_cache = {}; self.cache_hit_count = 0; self.cache_miss_count = 0; self._is_seeking = False
        if self.capture: self.capture.release(); self.capture = None
        try:
            temp_cap = cv2.VideoCapture(video_path)
            if temp_cap.isOpened():
                self.total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)); self.fps = temp_cap.get(cv2.CAP_PROP_FPS); self.width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                temp_cap.release()
                if not isinstance(self.fps, (int, float)) or self.fps <= 0:
                     print(f"Player WARN: Invalid FPS detected ({self.fps}), defaulting to 30.0")
                     self.fps = 30.0
                if not isinstance(self.total_frames, int) or self.total_frames <= 0:
                    print(f"Player WARN: Invalid total_frames detected ({self.total_frames}), defaulting to 0")
                    self.total_frames = 0
            else: self.total_frames = 0; self.fps = 0; self.width = 0; self.height = 0; print("Player WARN: cv2 failed to open video for metadata.")
        except Exception as e: print(f"Player Warning: Exception initializing video capture metadata: {str(e)}"); self.total_frames = 0; self.fps = 0; self.width = 0; self.height = 0
        if self.media_player: self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(video_path))))
        self._current_frame_internal = 0; self.last_emitted_frame = -1
        # Set timer interval based on FPS, aiming for slightly faster than frame rate
        ui_update_interval_ms = int(1000 / (self.fps * 1.5)) if self.fps > 0 else 33 # e.g., ~11ms for 60fps, default 33ms
        ui_update_interval_ms = max(10, ui_update_interval_ms) # Ensure minimum interval
        self.frame_timer.setInterval(ui_update_interval_ms); # print(f"Player: Frame timer interval set to {ui_update_interval_ms}ms for UI updates.") # Less verbose
        print(f"Player: Video info (initial): {self.width}x{self.height}, {self.fps:.2f} FPS, {self.total_frames} frames")
        # Force initial frame display slightly delayed to allow media loading
        QTimer.singleShot(100, lambda: self._force_update_frame(0)); self.clear_action(); return True

    def play(self):
        if not self.video_path or not self.media_player: return
        # Only start timer if player state actually becomes Playing
        # print("Player: Play command issued.") # Less verbose
        self.media_player.play()
        # Timer start is handled by _on_state_changed

    def pause(self):
        if not self.media_player: return
        # print("Player: Pause command issued.") # Less verbose
        self.media_player.pause()
        # Timer stop is handled by _on_state_changed
        # Force update on pause to ensure UI shows the exact pause frame
        QTimer.singleShot(20, self._update_frame_on_pause) # Delay slightly for position to stabilize

    def _update_frame_on_pause(self):
        """Force frame update after pausing."""
        if not self.is_playing: # Double check state
            current_pos = self.media_player.position()
            current_frame_on_pause = self._calculate_frame_from_position(current_pos)
            # print(f"Player: Forcing frame update on pause: F{current_frame_on_pause}") # Less verbose
            self._force_update_frame(current_frame_on_pause)

    def stop(self):
        if not self.media_player: return
        print("Player: Stop command issued.")
        self.media_player.stop();
        # Timer stop handled by _on_state_changed
        self._current_frame_internal = 0; self.last_emitted_frame = -1
        if self.capture: self.capture.release(); self.capture = None
        # Force display frame 0 on stop
        self._force_update_frame(0)

    def seek(self, frame):
        if not self.video_path or self.total_frames <= 0 or self.fps <= 0 or not self.media_player: return
        if self._is_seeking: return # Prevent recursive seeks

        self._is_seeking = True # Set flag
        target_frame = max(0, min(frame, self.total_frames - 1));
        position_ms = int((target_frame / self.fps) * 1000);
        was_playing = self.is_playing

        # print(f"Player: Seeking to F{target_frame} (Position {position_ms}ms)...") # Less verbose
        if was_playing:
            self.media_player.pause() # Pause before setting position
            # Note: Timer stop is handled by state change to Paused

        self.media_player.setPosition(position_ms)

        # Force update after a short delay to allow position to settle
        QTimer.singleShot(50, lambda f=target_frame: self._force_update_frame_after_seek(f, was_playing))


    def _force_update_frame_after_seek(self, frame_number, resume_playing):
        """Updates frame after seek delay and resumes play if needed."""
        # print(f"Player: Forcing frame update after seek: F{frame_number}") # Less verbose
        self._force_update_frame(frame_number)
        # Reset seek flag *after* update
        self._is_seeking = False
        # Resume playback if it was playing before seek
        if resume_playing:
            # print("Player: Resuming play after seek.") # Less verbose
            self.play()


    def _force_update_frame(self, frame_number):
        """Forces an update to a specific frame, bypassing the 'last_emitted_frame' check."""
        if not isinstance(self.total_frames, int) or self.total_frames <= 0:
            clamped_frame = 0
        else:
            clamped_frame = max(0, min(frame_number, self.total_frames - 1))

        qimage = self.get_qimage(clamped_frame)
        if qimage is not None and not qimage.isNull():
            # print(f"Player: Emitting FORCED frameChanged: F{clamped_frame}") # Less verbose
            self.frameChanged.emit(clamped_frame, qimage, self.current_action)
            self.last_emitted_frame = clamped_frame
        # else: # Debug
        #     print(f"Player WARN: Failed get QImage for FORCED update F{clamped_frame}")
        self._current_frame_internal = clamped_frame # Update internal estimate too


    def _get_cv2_capture(self):
        if not self.video_path: return None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened(): return cap
            else: print(f"Player Error: Failed to open temporary cv2 capture for {self.video_path}"); return None
        except Exception as e: print(f"Player Warning: Failed to open temporary cv2 capture: {e}"); return None

    def get_frame(self, frame_number):
        if self.total_frames <= 0 or frame_number < 0 or frame_number >= self.total_frames: return None
        if frame_number in self.frame_cache: self.cache_hit_count += 1; return self.frame_cache[frame_number]
        self.cache_miss_count += 1; rgb_frame = None; cap = self._get_cv2_capture()
        if not cap: return None
        try:
             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number); ret, frame = cap.read()
             if ret: rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); self.frame_cache[frame_number] = rgb_frame
             if len(self.frame_cache) > self.max_cache_size: oldest_frame = min(self.frame_cache.keys()); del self.frame_cache[oldest_frame]
        except Exception as e: print(f"Player Exception reading frame {frame_number}: {str(e)}")
        finally:
            if cap: cap.release()
        return rgb_frame

    def get_qimage(self, frame_number):
        frame = self.get_frame(frame_number);
        if frame is None: return None
        try:
            h, w, ch = frame.shape;
            if h <= 0 or w <= 0: return None
            bytes_per_line = ch * w; qimage = QImage(frame.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy(); return qimage
        except Exception as e: print(f"Player Exception converting frame {frame_number} to QImage: {str(e)}"); return None

    def _calculate_frame_from_position(self, position_ms):
        """Helper to calculate frame number from millisecond position."""
        if self.fps <= 0 or not isinstance(self.total_frames, int) or self.total_frames <= 0:
            return 0
        # Add a small offset to position calculation to handle potential rounding issues
        frame = int(((position_ms + 1) / 1000.0) * self.fps)
        return max(0, min(frame, self.total_frames - 1))

    # --- MODIFIED: _update_frame - Updates based on frame number change ---
    def _update_frame(self, target_frame_number):
        """Fetches and emits the target frame if it's different from the last emitted frame."""
        if not isinstance(self.total_frames, int) or self.total_frames <= 0:
            return # Cannot proceed without valid total_frames

        clamped_frame_number = max(0, min(target_frame_number, self.total_frames - 1))

        # --- Only proceed if the target frame is different from the last one successfully emitted ---
        if clamped_frame_number == self.last_emitted_frame:
            # print(f"Player DEBUG: Skipping update for F{clamped_frame_number}, same as last emitted.")
            return

        # --- Attempt to get the image ---
        qimage = self.get_qimage(clamped_frame_number)
        if qimage is not None and not qimage.isNull():
            # print(f"Player: Emitting frameChanged: F{clamped_frame_number}") # Less verbose
            self.frameChanged.emit(clamped_frame_number, qimage, self.current_action)
            self.last_emitted_frame = clamped_frame_number # Update last emitted ONLY on success
        # else: # Optional Debug
        #     print(f"Player WARN: Failed get QImage for F{clamped_frame_number}")

    # --- REMOVED: _on_position_changed - No longer directly used for frame updates ---

    # --- MODIFIED: _on_frame_timer - Now drives the update based on current position ---
    @pyqtSlot()
    def _on_frame_timer(self):
        """Called periodically during playback to update the frame based on player position."""
        if not self.is_playing or not self.media_player or self.fps <= 0 or self._is_seeking:
            return

        current_pos_ms = self.media_player.position()
        target_frame = self._calculate_frame_from_position(current_pos_ms)

        # --- COMMENTED OUT Print for Debugging ---
        # print(f"Player Timer Tick: Pos={current_pos_ms}ms -> Target F={target_frame} (Last Emitted: {self.last_emitted_frame})")

        # --- Update internal estimate & call the frame update logic ---
        self._current_frame_internal = target_frame
        self._update_frame(target_frame) # _update_frame handles the check for change


    @pyqtSlot(QMediaPlayer.State)
    def _on_state_changed(self, state):
        # print(f"Player Internal State Changed: {state}") # Less verbose
        if state == QMediaPlayer.PlayingState:
            self.is_playing = True
            if not self.frame_timer.isActive() and not self._is_seeking: # Don't start timer if seeking
                # print("Player: State=Playing, starting timer.") # Less verbose
                self.frame_timer.start()
        elif state == QMediaPlayer.PausedState:
            self.is_playing = False
            if self.frame_timer.isActive():
                # print("Player: State=Paused, stopping timer.") # Less verbose
                self.frame_timer.stop()
            # Trigger update on pause, unless seek is in progress
            if not self._is_seeking:
                 self._update_frame_on_pause()
        elif state == QMediaPlayer.StoppedState:
            self.is_playing = False
            if self.frame_timer.isActive():
                # print("Player: State=Stopped, stopping timer.") # Less verbose
                self.frame_timer.stop()
            self._current_frame_internal = 0
            self.last_emitted_frame = -1
            if self.capture: self.capture.release(); self.capture = None
            # Force update to frame 0 on stop
            self._force_update_frame(0)

    @pyqtSlot(QMediaPlayer.MediaStatus)
    def _on_media_status_changed(self, status):
        # print(f"Player Media Status Changed: {status}") # Less verbose
        if status == QMediaPlayer.EndOfMedia:
            print("Player: EndOfMedia status received.")
            self.is_playing = False
            if self.frame_timer.isActive(): self.frame_timer.stop()
            # Ensure the very last frame is displayed
            if self.total_frames > 0:
                last_frame_idx = self.total_frames - 1
                self._force_update_frame(last_frame_idx)
            # else: # Fallback if total_frames is unknown
            #     current_pos = self.media_player.position(); last_frame_idx = self._calculate_frame_from_position(current_pos); self._force_update_frame(last_frame_idx)
            if self.capture: self.capture.release(); self.capture = None
            self.videoFinished.emit(); # print("Player: videoFinished signal emitted.") # Less verbose
        elif status == QMediaPlayer.LoadedMedia:
             print("Player: LoadedMedia status received.")
             duration_ms = self.media_player.duration()
             if duration_ms > 0 and self.total_frames > 0 and self.fps <= 0 :
                  duration_sec = duration_ms / 1000.0; estimated_fps = self.total_frames / duration_sec
                  if 5 < estimated_fps < 120: print(f"Player: Estimating FPS from duration: {estimated_fps:.2f}"); self.fps = estimated_fps;
             elif duration_ms > 0 and self.fps > 0 and self.total_frames <= 0:
                  duration_sec = duration_ms / 1000.0; estimated_frames = int(duration_sec * self.fps)
                  if estimated_frames > 0: print(f"Player: Estimating frames from duration: {estimated_frames}"); self.total_frames = estimated_frames
             # Re-calculate timer interval based on potentially updated FPS
             ui_update_interval_ms = int(1000 / (self.fps * 1.5)) if self.fps > 0 else 33
             ui_update_interval_ms = max(10, ui_update_interval_ms)
             self.frame_timer.setInterval(ui_update_interval_ms)
             # print(f"Player: Frame timer interval updated to {ui_update_interval_ms}ms.") # Less verbose
             # Force update frame 0 after loading
             QTimer.singleShot(50, lambda: self._force_update_frame(0))
        elif status == QMediaPlayer.InvalidMedia: print("Player Error: InvalidMedia status received from QtMultimedia.")
        elif status == QMediaPlayer.BufferingMedia or status == QMediaPlayer.BufferedMedia: pass # Normal statuses
        elif status == QMediaPlayer.LoadingMedia: pass # print("Player: LoadingMedia status...") # Less verbose
        elif status == QMediaPlayer.StalledMedia: print("Player WARN: StalledMedia status received.")
        elif status == QMediaPlayer.NoMedia: print("Player: NoMedia status received.")
        # else: print(f"Player: Unhandled Media Status: {status}") # Less verbose


    def set_action(self, action_code): self.current_action = action_code;
    def clear_action(self): self.current_action = "";
    def get_video_properties(self):
        if (not isinstance(self.total_frames, int) or self.total_frames <= 0 or not isinstance(self.fps, (int,float)) or self.fps <= 0) and self.media_player and self.media_player.duration() > 0:
            print("Player: Triggering metadata update from get_video_properties.")
            self._on_media_status_changed(QMediaPlayer.LoadedMedia) # Try to update props if missing
        if not self.video_path: return None
        return { 'width': self.width, 'height': self.height, 'fps': self.fps, 'total_frames': self.total_frames }
    def get_video_widget(self): return self.container_widget
    def __del__(self):
        print("Player: Cleaning up media player resources (__del__ called)...")
        try:
            if self.frame_timer and self.frame_timer.isActive(): self.frame_timer.stop()
            if self.media_player:
                 self.media_player.stop(); self.media_player.setMedia(QMediaContent()); self.media_player.setVideoOutput(None)
            # Avoid deleting self.media_player if it's managed elsewhere or needed by Qt event loop
            # self.media_player = None # Maybe not safe depending on Qt lifecycle
        except RuntimeError as e: print(f"Player WARN: Expected RuntimeError during media player cleanup: {e}")
        except Exception as e: print(f"Player ERROR: Unexpected error during media player cleanup: {e}")
        if self.capture: self.capture.release(); self.capture = None

# --- END OF FILE qt_media_player.py ---