# qt_media_player.py - Enhanced video player with reliable audio and video playback
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QObject, pyqtSignal, QUrl, Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QFont
import cv2
import numpy as np
import os
import config

class QTMediaPlayer(QObject):
    """Enhanced media player with reliable audio/video playback."""
    frameChanged = pyqtSignal(int, QImage, str)
    videoFinished = pyqtSignal()
    
    def __init__(self):
        """Initialize the media player."""
        super().__init__()
        self.video_path = None
        self.media_player = QMediaPlayer()
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.current_action = ""
        self.capture = None
        
        # Create main container widget
        self.container_widget = QWidget()
        self.container_widget.setStyleSheet("background-color: black;")
        self.container_widget.setMinimumSize(768, 680)
        self.container_widget.setMaximumSize(768, 1300)
        
        # Use a simple layout for the video widget
        self.layout = QVBoxLayout(self.container_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Video widget for rendering
        self.video_widget = QVideoWidget()
        self.layout.addWidget(self.video_widget)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Frame timer for syncing to frames - adaptive to video framerate
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._on_frame_timer)
        
        # Connect signal for playback state changes
        self.media_player.stateChanged.connect(self._on_state_changed)
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.media_player.positionChanged.connect(self._on_position_changed)
        
        # Status flags
        self.is_playing = False
        self.is_updating_frame = False  # To prevent recursion
        self.last_emitted_frame = -1  # Track last emitted frame to avoid duplicates
        
        # Frame cache for seeking
        self.frame_cache = {}
        self.max_cache_size = 30  # Store at most this many frames
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    def set_video_path(self, video_path):
        """Set the video path and initialize the player."""
        print(f"Loading video: {video_path}")
        self.video_path = video_path

        # Clear frame cache
        self.frame_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Release previous resources
        if self.capture:
            self.capture.release()
            
        # Initialize video capture
        try:
            self.capture = cv2.VideoCapture(video_path)
            if not self.capture.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return False
                
            self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception as e:
            print(f"Exception initializing video capture: {str(e)}")
            self.capture = None
            return False

        # Set up media player
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(video_path))))
        self.current_frame = 0
        self.last_emitted_frame = -1

        # Set frame timer interval based on FPS - make it adaptive
        # Use half the frame interval for smoother updates
        frame_interval = max(10, int(500 / self.fps))  # At least 10ms, at most half frame time
        self.frame_timer.setInterval(frame_interval)

        print(f"Video loaded: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        print(f"Frame timer interval: {frame_interval}ms")

        # Grab and emit first frame
        self._update_frame(0)
        
        # Clear any existing action
        self.clear_action()

        return True
    
    def play(self):
        """Play the video."""
        if not self.video_path:
            return
            
        self.is_playing = True
        self.media_player.play()
        self.frame_timer.start()
    
    def pause(self):
        """Pause video playback."""
        self.is_playing = False
        self.media_player.pause()
        self.frame_timer.stop()
    
    def stop(self):
        """Stop video playback completely."""
        self.is_playing = False
        self.media_player.stop()
        self.frame_timer.stop()
        self.current_frame = 0
        self._update_frame(0)

    def seek(self, frame):
        """Seek to a specific frame."""
        if not self.video_path or not self.capture:
            return

        # Ensure frame is within valid range
        frame = max(0, min(frame, self.total_frames - 1))

        # Calculate position in milliseconds
        position = int((frame / self.fps) * 1000)

        # Remember playback state
        was_playing = self.is_playing

        # Pause playback during seek
        if was_playing:
            self.pause()

        # Seek to frame
        self.media_player.setPosition(position)
        self.current_frame = frame

        # Update frame display
        self._update_frame(frame)

        # Resume playback if needed
        if was_playing:
            self.play()

    def get_frame(self, frame_number):
        """Get a specific frame from the video (with caching)."""
        # Check if the frame is in the cache
        if frame_number in self.frame_cache:
            self.cache_hit_count += 1
            return self.frame_cache[frame_number]
            
        # Not in cache, need to read from file
        self.cache_miss_count += 1
        
        if not self.capture or not self.capture.isOpened():
            print("Warning: Video capture not initialized or closed")
            return None

        try:
            # Validate frame number
            if frame_number < 0 or frame_number >= self.total_frames:
                print(f"Warning: Frame number {frame_number} out of range (0-{self.total_frames - 1})")
                return None

            # Set position to specific frame
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the frame
            ret, frame = self.capture.read()
            if not ret:
                print(f"Warning: Failed to read frame {frame_number}")
                return None

            # Convert to RGB for Qt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add to cache
            self.frame_cache[frame_number] = rgb_frame
            
            # Limit cache size by removing oldest entries if needed
            if len(self.frame_cache) > self.max_cache_size:
                # Get the frame numbers sorted
                frames = sorted(self.frame_cache.keys())
                # Remove oldest frames
                for old_frame in frames[:len(frames) - self.max_cache_size]:
                    del self.frame_cache[old_frame]
                    
            return rgb_frame
        except Exception as e:
            print(f"Exception reading frame {frame_number}: {str(e)}")
            return None

    def get_qimage(self, frame_number):
        """Get QImage for specific frame."""
        frame = self.get_frame(frame_number)
        if frame is None:
            return None

        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            return QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        except Exception as e:
            print(f"Exception converting frame {frame_number} to QImage: {str(e)}")
            return None

    def _update_frame(self, frame_number):
        """Update the current frame display."""
        # Skip if this is the same frame we already emitted
        if frame_number == self.last_emitted_frame:
            return
            
        # Grab frame from the video for information purposes
        qimage = self.get_qimage(frame_number)

        # Only emit frame signal if we have a valid qimage
        if qimage is not None:
            # Emit frame signal with current state
            self.frameChanged.emit(frame_number, qimage, self.current_action)
            self.last_emitted_frame = frame_number
            self.current_frame = frame_number
        else:
            # Create a small empty QImage as a fallback
            empty_image = QImage(2, 2, QImage.Format_RGB888)
            empty_image.fill(Qt.black)
            self.frameChanged.emit(frame_number, empty_image, self.current_action)
            self.last_emitted_frame = frame_number
            self.current_frame = frame_number
            print(f"Warning: Could not get frame {frame_number}, using empty image")
    
    def _on_position_changed(self, position):
        """Handle media position change events."""
        if not self.is_playing or self.is_updating_frame:
            return
            
        # Calculate current frame based on position
        frame = int((position / 1000.0) * self.fps)
        
        # Only update if frame has changed
        if frame != self.current_frame and frame != self.last_emitted_frame:
            self.current_frame = frame
            self._update_frame(frame)

    def _on_frame_timer(self):
        """Handle frame timer events for precise frame tracking."""
        if not self.is_playing or self.is_updating_frame:
            return

        # Set flag to prevent recursion
        self.is_updating_frame = True

        try:
            # Calculate current position based on media player
            position = self.media_player.position()
            new_frame = int((position / 1000.0) * self.fps)

            # Only update if frame has changed and is valid
            if new_frame != self.current_frame and 0 <= new_frame < self.total_frames:
                old_frame = self.current_frame
                self.current_frame = new_frame

                # If we've skipped more than one frame, ensure we update each
                # This prevents missing frames during annotation
                if new_frame - old_frame > 1:
                    #print(f"Detected frame skip: {old_frame} to {new_frame}")
                    # Just update the current frame - range-based tracking handles the rest
                    self._update_frame(new_frame)
                else:
                    # Normal sequential frame update
                    self._update_frame(new_frame)
        finally:
            # Clear flag when done
            self.is_updating_frame = False
    
    def _on_state_changed(self, state):
        """Handle media player state changes."""
        if state == QMediaPlayer.StoppedState:
            self.is_playing = False
            self.frame_timer.stop()
            self.videoFinished.emit()
    
    def _on_media_status_changed(self, status):
        """Handle media status changes."""
        if status == QMediaPlayer.EndOfMedia:
            self.is_playing = False
            self.frame_timer.stop()
            self.videoFinished.emit()

    def set_action(self, action_code):
        """Set the current action code."""
        # Just store the action
        if self.current_action != action_code:
            self.current_action = action_code

            # Only emit with a valid image
            qimage = self.get_qimage(self.current_frame)
            if qimage is not None:
                # Emit frame changed with the new action code
                self.frameChanged.emit(self.current_frame, qimage, action_code)
            else:
                # Create a small empty QImage as a fallback
                empty_image = QImage(2, 2, QImage.Format_RGB888)
                empty_image.fill(Qt.black)
                self.frameChanged.emit(self.current_frame, empty_image, action_code)
                print(f"Warning: Could not get frame {self.current_frame} for action display, using empty image")

    def clear_action(self):
        """Clear the current action code."""
        if self.current_action:
            self.current_action = ""

            # Only emit with a valid image
            qimage = self.get_qimage(self.current_frame)
            if qimage is not None:
                # Emit frame changed with empty action code
                self.frameChanged.emit(self.current_frame, qimage, "")
            else:
                # Create a small empty QImage as a fallback
                empty_image = QImage(2, 2, QImage.Format_RGB888)
                empty_image.fill(Qt.black)
                self.frameChanged.emit(self.current_frame, empty_image, "")
                print(f"Warning: Could not get frame {self.current_frame} for action display, using empty image")
    
    def get_video_properties(self):
        """Get video properties."""
        if not self.video_path:
            return None
        
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames
        }
    
    def get_video_widget(self):
        """Get the container widget for embedding in UI."""
        return self.container_widget
