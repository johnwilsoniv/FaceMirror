# qt_media_player.py - Enhanced video player with reliable audio and video playback
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QUrl, Qt, QTimer, QThread
from PyQt5.QtGui import QImage
import cv2
import os
import time
import config
from collections import OrderedDict  # For proper LRU cache implementation
import threading  # For thread-safe cache access

# Diagnostic profiling (only enabled if flag is set in config)
def _get_diagnostic_profiler_if_enabled():
    """Get diagnostic profiler only if enabled in config"""
    try:
        import config
        if not config.ENABLE_DIAGNOSTIC_PROFILING:
            return None
        from diagnostic_profiler import get_diagnostic_profiler
        return get_diagnostic_profiler()
    except (ImportError, AttributeError):
        return None


class FramePreloader(QThread):
    """
    Background thread that pre-decodes frames ahead of current playback position.
    Dramatically improves playback smoothness by warming the cache.
    """

    # Signal emitted when frames are preloaded
    frames_preloaded = pyqtSignal(int)  # number of frames preloaded

    def __init__(self, player):
        super().__init__()
        self.player = player
        self.running = False
        self.paused = True
        self._lock = threading.Lock()

        # Preloader configuration
        self.preload_ahead_count = 30  # Number of frames to preload ahead
        self.check_interval_ms = 100   # Check every 100ms

    def set_preload_count(self, count):
        """Set how many frames to preload ahead"""
        with self._lock:
            self.preload_ahead_count = count

    def resume(self):
        """Resume preloading (called when playback starts)"""
        with self._lock:
            self.paused = False

    def pause(self):
        """Pause preloading (called when playback pauses)"""
        with self._lock:
            self.paused = True

    def stop(self):
        """Stop the preloader thread"""
        self.running = False
        self.wait()

    def run(self):
        """Main preloader loop - runs in background thread"""
        self.running = True

        while self.running:
            # Sleep for check interval
            self.msleep(self.check_interval_ms)

            # Skip if paused or player not ready
            with self._lock:
                if self.paused:
                    continue

            if not self.player or not self.player.capture or self.player.total_frames <= 0:
                continue

            try:
                # Get current playback position
                current_frame = self.player.current_frame
                if current_frame < 0:
                    current_frame = 0

                # Calculate range of frames to preload
                start_frame = current_frame + 1
                end_frame = min(current_frame + self.preload_ahead_count, self.player.total_frames)

                preloaded_count = 0

                # Preload frames that aren't already cached
                for frame_num in range(start_frame, end_frame):
                    # Check if still running and not paused
                    with self._lock:
                        if not self.running or self.paused:
                            break

                    # Skip if already in cache (thread-safe check)
                    with self.player.cache_lock:
                        already_cached = frame_num in self.player.frame_cache
                        cache_full = len(self.player.frame_cache) >= self.player.max_cache_size

                    if already_cached:
                        continue

                    # Check cache size limit before adding more
                    if cache_full:
                        # Cache is full - stop preloading
                        break

                    # Preload this frame (will add to cache)
                    # Use the player's get_frame method which handles caching and locking
                    frame = self.player.get_frame(frame_num)
                    if frame is not None:
                        preloaded_count += 1

                # Emit signal if we preloaded any frames
                if preloaded_count > 0:
                    self.frames_preloaded.emit(preloaded_count)

            except Exception as e:
                print(f"FramePreloader ERROR: {e}")
                import traceback
                traceback.print_exc()


class QTMediaPlayer(QObject):
    """Enhanced media player with reliable audio/video playback."""
    frameChanged = pyqtSignal(int, QImage, str) # frame_num, qimage, action_code
    videoFinished = pyqtSignal()
    audioExtractionError = pyqtSignal(str)

    def __init__(self):
        super().__init__(); self.video_path = None; self.media_player = QMediaPlayer(); self.total_frames = 0
        self._current_frame_internal = 0; self.fps = 0; self.width = 0; self.height = 0; self.current_action = ""
        self.capture = None  # Persistent OpenCV VideoCapture for hardware-accelerated frame extraction
        self._hw_accel_enabled = False  # Track if hardware acceleration was successfully enabled
        self.container_widget = QWidget(); self.container_widget.setStyleSheet("background-color: black;")
        self.container_widget.setMinimumSize(640, 480); self.layout = QVBoxLayout(self.container_widget)
        self.layout.setContentsMargins(0, 0, 0, 0); self.layout.setSpacing(0); self.video_widget = QVideoWidget()
        self.layout.addWidget(self.video_widget);
        if self.media_player: self.media_player.setVideoOutput(self.video_widget)
        self.frame_timer = QTimer(); self.frame_timer.setTimerType(Qt.PreciseTimer); self.frame_timer.timeout.connect(self._on_frame_timer)
        if self.media_player:
            self.media_player.stateChanged.connect(self._on_state_changed); self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
            # We rely less on positionChanged now, using timer + getPosition instead
            # self.media_player.positionChanged.connect(self._on_position_changed)
        self.is_playing = False; self.last_emitted_frame = -1
        # Proper LRU cache with QImage caching
        self.frame_cache = OrderedDict()
        self.qimage_cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.max_cache_size = 50
        self.max_qimage_cache_size = 25
        self.cache_hit_count = 0; self.cache_miss_count = 0
        # Performance profiling
        self._frame_extraction_times = []
        self._max_profile_samples = 100
        # Prevent recursive updates during seek
        self._is_seeking = False
        # Position interpolation tracking
        self._last_confirmed_position_ms = 0
        self._last_position_update_time = 0
        self._last_resync_time = 0
        # Diagnostic profiler
        self.diagnostic_profiler = _get_diagnostic_profiler_if_enabled()
        # Background frame preloader
        self.frame_preloader = FramePreloader(self)
        self.frame_preloader.start()  # Start the background thread (paused initially)

    @property
    def current_frame(self):
        # Return the last frame we successfully emitted, otherwise the internal estimate
        return self.last_emitted_frame if self.last_emitted_frame >= 0 else self._current_frame_internal

    def set_video_path(self, video_path):
        print(f"Player: Loading video: {video_path}")
        try:
            if self.media_player:
                self.media_player.stop() # Stop existing playback/timer
            if not os.path.exists(video_path):
                print(f"Player Error: Video file does not exist at {video_path}")
                return False

            self.video_path = video_path
            # Clear both caches on new video
            self.frame_cache.clear()
            self.qimage_cache.clear()
            self.cache_hit_count = 0
            self.cache_miss_count = 0
            self._is_seeking = False
            self._frame_extraction_times = []  # Reset profiling on new video

            # Release old capture if exists - with extra safety
            if self.capture:
                try:
                    self.capture.release()
                    print("Player: Released previous VideoCapture")
                except Exception as e:
                    print(f"Player WARNING: Error releasing previous VideoCapture: {e}")
                finally:
                    self.capture = None

            # Small delay to ensure resources are fully released
            import time
            time.sleep(0.05)  # 50ms delay

        # Create persistent VideoCapture with best available backend
        except Exception as e:
            print(f"Player ERROR: Exception during video setup: {e}")
            import traceback
            traceback.print_exc()
            return False

        try:
            # On macOS, AVFoundation provides the fastest software decoding
            # Note: Hardware acceleration (VideoToolbox) is not available through OpenCV's
            # CAP_PROP_HW_ACCELERATION API on macOS - only Windows/Linux are supported
            print("Player: Creating persistent VideoCapture with AVFoundation backend...")
            self.capture = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)

            if self.capture and self.capture.isOpened():
                print("Player: AVFoundation VideoCapture created successfully (software decode)")
                self._hw_accel_enabled = False  # Software decode only on macOS

                # Get video properties from persistent capture
                self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.capture.get(cv2.CAP_PROP_FPS)
                self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if not isinstance(self.fps, (int, float)) or self.fps <= 0:
                     print(f"Player WARN: Invalid FPS detected ({self.fps}), defaulting to 30.0")
                     self.fps = 30.0
                if not isinstance(self.total_frames, int) or self.total_frames <= 0:
                    print(f"Player WARN: Invalid total_frames detected ({self.total_frames}), defaulting to 0")
                    self.total_frames = 0

                # === MEMORY OPTIMIZATION: Adaptive cache sizing based on frame size ===
                # Calculate memory per frame (width * height * 3 bytes for RGB)
                bytes_per_frame = self.width * self.height * 3
                mb_per_frame = bytes_per_frame / (1024 * 1024)

                # Target cache size: INCREASED for better scrubbing performance
                # 500MB RGB frames allows caching more frames for back/forth scrubbing
                # 250MB QImage cache reduces conversion overhead
                target_rgb_cache_mb = 500     # Was 100 - 5x increase for better scrubbing
                target_qimage_cache_mb = 250  # Was 50 - 5x increase for better scrubbing

                # Calculate max frames for each cache
                self.max_cache_size = max(10, int(target_rgb_cache_mb / mb_per_frame))
                self.max_qimage_cache_size = max(5, int(target_qimage_cache_mb / mb_per_frame))
            else:
                # Fallback to FFmpeg backend
                print("Player: AVFoundation failed, falling back to FFmpeg backend...")
                if self.capture: self.capture.release()
                self.capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

                if self.capture and self.capture.isOpened():
                    print("Player: FFmpeg VideoCapture created (software decode)")
                    self._hw_accel_enabled = False
                    self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.fps = self.capture.get(cv2.CAP_PROP_FPS)
                    self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    self.total_frames = 0; self.fps = 0; self.width = 0; self.height = 0
                    self._hw_accel_enabled = False
                    print("Player WARN: cv2 failed to open video even with fallback.")
        except Exception as e:
            print(f"Player Warning: Exception initializing persistent VideoCapture: {str(e)}")
            import traceback
            traceback.print_exc()
            self.total_frames = 0; self.fps = 0; self.width = 0; self.height = 0
            self._hw_accel_enabled = False

        # Set Qt media player content
        try:
            if self.media_player:
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(video_path))))
                print("Player: Qt media player content set successfully")
        except Exception as e:
            print(f"Player ERROR: Exception setting media player content: {e}")
            import traceback
            traceback.print_exc()

        self._current_frame_internal = 0; self.last_emitted_frame = -1
        # Set timer interval to match display refresh rate for smooth UI updates
        ui_update_interval_ms = config.UI_UPDATE_INTERVAL_MS
        self.frame_timer.setInterval(ui_update_interval_ms)
        # Initialize position interpolation tracking
        self._last_confirmed_position_ms = 0
        self._last_position_update_time = time.time()
        self._last_resync_time = time.time()
        # Force initial frame display with longer delay to ensure media is fully loaded
        # Also seek to frame 0 to ensure QMediaPlayer is positioned at the start
        QTimer.singleShot(200, self._load_initial_frame); self.clear_action(); return True

    def _load_initial_frame(self):
        """Load and display the first frame after video loads."""
        # To display frame 0 on QVideoWidget, we need to play and immediately pause
        # QVideoWidget only shows frames when QMediaPlayer has been activated
        if self.media_player:
            self.media_player.setPosition(0)
            # Play briefly to load frame 0 into QVideoWidget
            self.media_player.play()
            # Pause immediately (after 50ms to allow frame to load)
            QTimer.singleShot(50, self.media_player.pause)
        # Also emit frame update signal for timeline/UI
        QTimer.singleShot(100, lambda: self._force_update_frame(0))

    def play(self):
        if not self.video_path or not self.media_player: return
        # Only start timer if player state actually becomes Playing
        self.media_player.play()
        # Resume frame preloader for smooth playback
        if self.frame_preloader:
            self.frame_preloader.resume()
        # Timer start is handled by _on_state_changed

    def pause(self):
        if not self.media_player: return
        self.media_player.pause()
        # Pause frame preloader
        if self.frame_preloader:
            self.frame_preloader.pause()
        # Timer stop is handled by _on_state_changed
        # Force update on pause to ensure UI shows the exact pause frame
        QTimer.singleShot(20, self._update_frame_on_pause) # Delay slightly for position to stabilize

    def _update_frame_on_pause(self):
        """Force frame update after pausing."""
        if not self.is_playing: # Double check state
            current_pos = self.media_player.position()
            current_frame_on_pause = self._calculate_frame_from_position(current_pos)
            self._force_update_frame(current_frame_on_pause)

    def stop(self):
        if not self.media_player: return
        # Pause frame preloader
        if self.frame_preloader:
            self.frame_preloader.pause()
        # Print performance stats before cleanup
        if self._frame_extraction_times:
            self.print_performance_stats()
        self.media_player.stop();
        # Timer stop handled by _on_state_changed
        self._current_frame_internal = 0; self.last_emitted_frame = -1
        # Note: Keep capture open for potential reuse, only release on new video load
        # Force display frame 0 on stop
        self._force_update_frame(0)

    def seek(self, frame):
        if not self.video_path or self.total_frames <= 0 or self.fps <= 0 or not self.media_player: return
        if self._is_seeking: return # Prevent recursive seeks

        self._is_seeking = True # Set flag
        target_frame = max(0, min(frame, self.total_frames - 1));
        position_ms = int((target_frame / self.fps) * 1000);
        was_playing = self.is_playing

        # OPTIMIZATION: Pause preloader during seek to prioritize user input
        # Prevents preloader from competing for VideoCapture lock
        if self.frame_preloader:
            self.frame_preloader.pause()

        if was_playing:
            self.media_player.pause() # Pause before setting position
            # Note: Timer stop is handled by state change to Paused

        self.media_player.setPosition(position_ms)

        # Force update after a short delay to allow position to settle
        QTimer.singleShot(50, lambda f=target_frame: self._force_update_frame_after_seek(f, was_playing))


    def _force_update_frame_after_seek(self, frame_number, resume_playing):
        """Updates frame after seek delay and resumes play if needed."""
        # Reset interpolation tracking after seek to prevent jumps
        current_pos = self.media_player.position()
        self._last_confirmed_position_ms = current_pos
        self._last_position_update_time = time.time()
        self._last_resync_time = time.time()
        self._force_update_frame(frame_number)
        # Reset seek flag *after* update
        self._is_seeking = False
        # Resume playback if it was playing before seek
        if resume_playing:
            self.play()  # This will resume the preloader
        # If not resuming playback, keep preloader paused (user is scrubbing)


    def _force_update_frame(self, frame_number):
        """Forces an update to a specific frame, bypassing the 'last_emitted_frame' check."""
        if not isinstance(self.total_frames, int) or self.total_frames <= 0:
            clamped_frame = 0
        else:
            clamped_frame = max(0, min(frame_number, self.total_frames - 1))

        qimage = self.get_qimage(clamped_frame)
        if qimage is not None and not qimage.isNull():
            self.frameChanged.emit(clamped_frame, qimage, self.current_action)
            self.last_emitted_frame = clamped_frame
        self._current_frame_internal = clamped_frame


    def get_frame(self, frame_number):
        """Extract frame using persistent hardware-accelerated VideoCapture with LRU caching"""
        if self.total_frames <= 0 or frame_number < 0 or frame_number >= self.total_frames: return None

        # Check cache first (thread-safe)
        with self.cache_lock:
            if frame_number in self.frame_cache:
                self.cache_hit_count += 1
                # Record cache hit in diagnostic profiler
                if self.diagnostic_profiler:
                    self.diagnostic_profiler.record_cache_hit('rgb', time_saved_ms=15)  # Estimated time saved
                # Move to end (most recently used)
                self.frame_cache.move_to_end(frame_number)
                return self.frame_cache[frame_number]

            self.cache_miss_count += 1
            # Record cache miss in diagnostic profiler
            if self.diagnostic_profiler:
                self.diagnostic_profiler.record_cache_miss('rgb')

        rgb_frame = None

        # Check if persistent capture is available
        if not self.capture or not self.capture.isOpened():
            print(f"Player Error: Persistent capture not available for frame {frame_number}")
            return None

        # Performance profiling start
        start_time = time.time()

        try:
            # === CRITICAL: VideoCapture operations must be thread-safe ===
            # OpenCV's VideoCapture is NOT thread-safe - lock all access
            with self.capture_lock:
                # Track seek operation
                seek_start = time.time()
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                if self.diagnostic_profiler:
                    self.diagnostic_profiler.record_timing('video', 'frame_seek', (time.time() - seek_start) * 1000)

                # Track frame read operation
                read_start = time.time()
                ret, frame = self.capture.read()
                if self.diagnostic_profiler:
                    self.diagnostic_profiler.record_timing('video', 'frame_read', (time.time() - read_start) * 1000)

            # Color conversion (outside lock - doesn't need VideoCapture)
            if ret:
                # Track color conversion
                convert_start = time.time()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.diagnostic_profiler:
                    self.diagnostic_profiler.record_timing('video', 'color_conversion', (time.time() - convert_start) * 1000)

                # Add to cache (thread-safe)
                with self.cache_lock:
                    self.frame_cache[frame_number] = rgb_frame

                    # Proper LRU eviction: remove oldest (first) item when cache is full
                    if len(self.frame_cache) > self.max_cache_size:
                        # popitem(last=False) removes the first (oldest) item
                        self.frame_cache.popitem(last=False)
        except Exception as e:
            print(f"Player Exception reading frame {frame_number}: {str(e)}")

        # Performance profiling end
        extraction_time = (time.time() - start_time) * 1000  # Convert to ms
        self._frame_extraction_times.append(extraction_time)

        # Keep only last N samples
        if len(self._frame_extraction_times) > self._max_profile_samples:
            self._frame_extraction_times.pop(0)

        # Record total extraction time in diagnostic profiler
        if self.diagnostic_profiler:
            self.diagnostic_profiler.record_timing('video', 'total_frame_extraction', extraction_time)

        return rgb_frame

    def get_qimage(self, frame_number):
        """Get QImage with two-level caching: QImage cache + RGB frame cache"""
        # Check QImage cache first (thread-safe, avoids conversion overhead)
        with self.cache_lock:
            if frame_number in self.qimage_cache:
                # Record cache hit in diagnostic profiler
                if self.diagnostic_profiler:
                    self.diagnostic_profiler.record_cache_hit('qimage', time_saved_ms=5)  # Estimated time saved
                # Move to end (most recently used)
                self.qimage_cache.move_to_end(frame_number)
                return self.qimage_cache[frame_number]

            # Record cache miss in diagnostic profiler
            if self.diagnostic_profiler:
                self.diagnostic_profiler.record_cache_miss('qimage')

        # QImage cache miss - get RGB frame (may hit frame cache)
        frame = self.get_frame(frame_number)
        if frame is None: return None

        try:
            # Track QImage conversion time
            convert_start = time.time()

            h, w, ch = frame.shape
            if h <= 0 or w <= 0: return None
            bytes_per_line = ch * w

            # OPTIMIZATION: Create QImage with copy of data to avoid lifetime issues
            # The copy() ensures Qt owns the data and won't crash on numpy array cleanup
            qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            if self.diagnostic_profiler:
                self.diagnostic_profiler.record_timing('video', 'qimage_conversion', (time.time() - convert_start) * 1000)

            # Cache the QImage for reuse (thread-safe, avoids repeated conversion)
            with self.cache_lock:
                self.qimage_cache[frame_number] = qimage

                # Manage QImage cache size (LRU eviction)
                if len(self.qimage_cache) > self.max_qimage_cache_size:
                    self.qimage_cache.popitem(last=False)

            return qimage
        except Exception as e:
            print(f"Player Exception converting frame {frame_number} to QImage: {str(e)}")
            return None

    def get_performance_stats(self):
        """Get frame extraction performance statistics"""
        if not self._frame_extraction_times:
            return {
                'hw_accel': self._hw_accel_enabled,
                'avg_ms': 0,
                'min_ms': 0,
                'max_ms': 0,
                'samples': 0,
                'cache_hits': self.cache_hit_count,
                'cache_misses': self.cache_miss_count
            }

        return {
            'hw_accel': self._hw_accel_enabled,
            'avg_ms': sum(self._frame_extraction_times) / len(self._frame_extraction_times),
            'min_ms': min(self._frame_extraction_times),
            'max_ms': max(self._frame_extraction_times),
            'samples': len(self._frame_extraction_times),
            'cache_hits': self.cache_hit_count,
            'cache_misses': self.cache_miss_count,
            'cache_hit_rate': self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100 if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        }

    def print_performance_stats(self):
        """Print performance statistics to console"""
        stats = self.get_performance_stats()
        print(f"\n{'='*60}")
        print(f"Frame Extraction Performance Statistics")
        print(f"{'='*60}")
        print(f"Backend: AVFoundation (software decode)")
        print(f"Note: Hardware acceleration unavailable on macOS via OpenCV")
        print(f"Frame Extraction Times (ms):")
        print(f"  Average: {stats['avg_ms']:.2f}ms")
        print(f"  Min:     {stats['min_ms']:.2f}ms")
        print(f"  Max:     {stats['max_ms']:.2f}ms")
        print(f"  Samples: {stats['samples']}")
        print(f"Cache Statistics:")
        print(f"  Hits:    {stats['cache_hits']}")
        print(f"  Misses:  {stats['cache_misses']}")
        print(f"  Hit Rate: {stats.get('cache_hit_rate', 0):.1f}%")
        print(f"{'='*60}\n")

    def _calculate_frame_from_position(self, position_ms):
        """Helper to calculate frame number from millisecond position."""
        if self.fps <= 0 or not isinstance(self.total_frames, int) or self.total_frames <= 0:
            return 0
        # Add a small offset to position calculation to handle potential rounding issues
        frame = int(((position_ms + 1) / 1000.0) * self.fps)
        return max(0, min(frame, self.total_frames - 1))

    def _update_frame(self, target_frame_number):
        """Fetches and emits the target frame if it's different from the last emitted frame."""
        if not isinstance(self.total_frames, int) or self.total_frames <= 0:
            return

        clamped_frame_number = max(0, min(target_frame_number, self.total_frames - 1))

        if clamped_frame_number == self.last_emitted_frame:
            return
        qimage = self.get_qimage(clamped_frame_number)
        if qimage is not None and not qimage.isNull():
            self.frameChanged.emit(clamped_frame_number, qimage, self.current_action)
            self.last_emitted_frame = clamped_frame_number
    @pyqtSlot()
    def _on_frame_timer(self):
        """Called periodically during playback to update the frame based on interpolated position."""
        if not self.is_playing or not self.media_player or self.fps <= 0 or self._is_seeking:
            return

        current_time = time.time()

        # Determine if we need to re-sync with actual QMediaPlayer position
        time_since_last_resync = (current_time - self._last_resync_time) * 1000  # Convert to ms
        should_resync = time_since_last_resync >= config.POSITION_RESYNC_INTERVAL_MS

        if should_resync:
            # Get actual position from QMediaPlayer to prevent drift
            actual_pos_ms = self.media_player.position()
            self._last_confirmed_position_ms = actual_pos_ms
            self._last_position_update_time = current_time
            self._last_resync_time = current_time
            estimated_pos_ms = actual_pos_ms
        else:
            # Interpolate position based on elapsed time since last update
            elapsed_time_ms = (current_time - self._last_position_update_time) * 1000
            # Assume normal playback speed (1.0x)
            estimated_pos_ms = self._last_confirmed_position_ms + elapsed_time_ms

        # Convert estimated position to frame number
        target_frame = self._calculate_frame_from_position(int(estimated_pos_ms))

        # Update internal estimate & call the frame update logic
        self._current_frame_internal = target_frame
        self._update_frame(target_frame)  # _update_frame handles the check for change


    @pyqtSlot(QMediaPlayer.State)
    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.is_playing = True
            # Reset interpolation tracking when playback starts
            current_pos = self.media_player.position()
            self._last_confirmed_position_ms = current_pos
            self._last_position_update_time = time.time()
            self._last_resync_time = time.time()
            if not self.frame_timer.isActive() and not self._is_seeking: # Don't start timer if seeking
                self.frame_timer.start()
        elif state == QMediaPlayer.PausedState:
            self.is_playing = False
            if self.frame_timer.isActive():
                self.frame_timer.stop()
            # Trigger update on pause, unless seek is in progress
            if not self._is_seeking:
                 self._update_frame_on_pause()
        elif state == QMediaPlayer.StoppedState:
            self.is_playing = False
            if self.frame_timer.isActive():
                self.frame_timer.stop()
            self._current_frame_internal = 0
            self.last_emitted_frame = -1
            # Keep capture open for potential replay/reuse (same as EndOfMedia handling)
            # Force update to frame 0 on stop
            self._force_update_frame(0)

    @pyqtSlot(QMediaPlayer.MediaStatus)
    def _on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            print("Player: EndOfMedia status received.")
            # Print performance statistics
            if self._frame_extraction_times:
                self.print_performance_stats()
            self.is_playing = False
            if self.frame_timer.isActive(): self.frame_timer.stop()
            # Ensure the very last frame is displayed
            if self.total_frames > 0:
                last_frame_idx = self.total_frames - 1
                self._force_update_frame(last_frame_idx)
            # Keep capture open for potential replay/reuse (no longer releasing here)
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
             # Timer interval is based on display refresh rate, not video FPS
             # (already set in load_video, no need to recalculate)
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
            # Stop frame preloader
            if hasattr(self, 'frame_preloader') and self.frame_preloader:
                self.frame_preloader.stop()

            if self.frame_timer and self.frame_timer.isActive(): self.frame_timer.stop()
            if self.media_player:
                 self.media_player.stop(); self.media_player.setMedia(QMediaContent()); self.media_player.setVideoOutput(None)
            # Avoid deleting self.media_player if it's managed elsewhere or needed by Qt event loop
            # self.media_player = None # Maybe not safe depending on Qt lifecycle
        except RuntimeError as e: print(f"Player WARN: Expected RuntimeError during media player cleanup: {e}")
        except Exception as e: print(f"Player ERROR: Unexpected error during media player cleanup: {e}")
        if self.capture: self.capture.release(); self.capture = None

