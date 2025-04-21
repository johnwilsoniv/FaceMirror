# --- START OF FILE playback_manager.py ---

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtMultimedia import QMediaPlayer # *** ADDED IMPORT ***

class PlaybackManager(QObject):
    """Manages the QTMediaPlayer instance and playback state."""

    # Signals to Controller/UIManager
    playback_state_changed = pyqtSignal(bool) # True if playing, False if paused/stopped
    frame_update_needed = pyqtSignal(int, QImage) # Emits frame number and image for UI/processing
    video_ended = pyqtSignal()
    playback_error = pyqtSignal(str) # For critical player errors

    def __init__(self, player, parent=None):
        super().__init__(parent)
        self.player = player
        if not self.player:
            raise ValueError("PlaybackManager requires a valid QTMediaPlayer instance.")

        self._connect_player_signals()
        self._is_paused_for_prompt = False # Internal flag

    def _connect_player_signals(self):
        """Connect signals FROM the player instance."""
        self.player.frameChanged.connect(self._on_player_frame_changed)
        self.player.videoFinished.connect(self._on_player_video_finished)
        # Connect player's internal state changes
        self.player.media_player.stateChanged.connect(self._on_player_state_changed) # *** Connect here ***

    @property
    def is_playing(self):
        return self.player.is_playing

    @property
    def current_frame(self):
        return self.player.current_frame

    @property
    def total_frames(self):
        return self.player.total_frames

    @property
    def fps(self):
        return self.player.fps

    def get_video_properties(self):
        return self.player.get_video_properties()

    # --- Slots for Controlling Playback ---

    @pyqtSlot(bool)
    def set_paused_for_prompt(self, is_paused):
        """Informs the manager if the pause is due to a user prompt."""
        self._is_paused_for_prompt = is_paused

    @pyqtSlot(str)
    def load_video(self, video_path):
        """Loads a new video into the player."""
        print(f"PlaybackManager: Loading video {video_path}")
        # Stop current playback before loading
        if self.is_playing:
             self.pause()
        # Reset prompt flag on new video load
        self._is_paused_for_prompt = False
        success = self.player.set_video_path(video_path)
        if not success:
            self.playback_error.emit(f"Failed to load video: {video_path}")
        return success

    @pyqtSlot()
    def play(self):
        """Starts or resumes playback."""
        if self._is_paused_for_prompt:
            print("PlaybackManager: Play requested but paused for prompt. Ignoring.")
            return
        if self.player.video_path and not self.is_playing:
            # print("PlaybackManager: Requesting Play") # Less verbose
            self.player.play()
            # self.playback_state_changed.emit(True) # State change handled by _on_player_state_changed

    @pyqtSlot()
    def pause(self, triggered_by_prompt=False):
        """Pauses playback."""
        if self.is_playing:
            # print(f"PlaybackManager: Requesting Pause (Prompt: {triggered_by_prompt})") # Less verbose
            self.player.pause()
            self.set_paused_for_prompt(triggered_by_prompt) # Set flag *after* pausing


    @pyqtSlot(int)
    def seek(self, frame_number):
        """Seeks to a specific frame."""
        if self.player.video_path:
            # print(f"PlaybackManager: Seeking to frame {frame_number}") # Less verbose
            # Player's seek method already handles pausing/resuming if needed
            self.player.seek(frame_number)
            # Frame update will be triggered by player's internal signals after seek settles

    # --- Slots for Handling Player Signals ---

    @pyqtSlot(int, QImage, str) # Ignore action_code from player signal for now
    def _on_player_frame_changed(self, frame_number, qimage, _):
        """Relays frame updates."""
        # --- Add Print ---
        # print(f"PlaybackManager: Relaying frame_update_needed for F{frame_number}") # Commented out verbose log
        self.frame_update_needed.emit(frame_number, qimage)

    @pyqtSlot()
    def _on_player_video_finished(self):
        """Relays video finished signal."""
        print("PlaybackManager: Video finished.")
        self.playback_state_changed.emit(False) # Ensure state is updated
        self._is_paused_for_prompt = False # Reset flag
        self.video_ended.emit()

    @pyqtSlot(QMediaPlayer.State)
    def _on_player_state_changed(self, state):
        """Handles internal player state changes."""
        is_currently_playing = (state == QMediaPlayer.PlayingState)
        # print(f"PlaybackManager: Player state changed to {state}. Emitting playback_state_changed({is_currently_playing}).") # Less verbose
        self.playback_state_changed.emit(is_currently_playing)
        # Reset prompt flag if stopped
        if state == QMediaPlayer.StoppedState:
             self._is_paused_for_prompt = False

# --- END OF FILE playback_manager.py ---