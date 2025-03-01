# player_integration.py - Helper module to integrate the enhanced video player
from PyQt5.QtWidgets import QVBoxLayout

def integrate_qt_player(main_window, media_player):
    """
    Integrates the QT media player into the main window UI.
    
    Args:
        main_window: The MainWindow instance from gui_component.py
        media_player: The QTMediaPlayer instance
    """
    # Get the video widget from the media player
    video_widget = media_player.get_video_widget()
    
    # Get the video layout from the main window
    video_layout = main_window.video_label.parentWidget().layout()
    
    # Remove the old video label
    video_layout.removeWidget(main_window.video_label)
    main_window.video_label.hide()
    
    # Insert the video widget at the same position
    video_layout.insertWidget(0, video_widget)
    
    # Make the video widget visible
    video_widget.show()
    
    # Store references to prevent garbage collection
    main_window.video_widget = video_widget
    main_window.media_player = media_player


def replace_video_player(app_controller):
    """
    Replaces the OpenCVVideoPlayer with the new QTMediaPlayer.
    
    Args:
        app_controller: The ApplicationController instance from main.py
    
    Returns:
        The new QTMediaPlayer instance
    """
    from qt_media_player import QTMediaPlayer
    
    # Create new media player
    new_player = QTMediaPlayer()
    
    # Disconnect signals from old player
    app_controller.window.play_pause_signal.disconnect(app_controller.toggle_play_pause)
    app_controller.window.frame_changed_signal.disconnect(app_controller.seek_to_frame)
    app_controller.video_player.frameChanged.disconnect(app_controller.update_frame)
    app_controller.video_player.videoFinished.disconnect(app_controller.video_finished)
    
    # Store old player for reference
    old_player = app_controller.video_player
    
    # Replace player reference
    app_controller.video_player = new_player
    
    # Reconnect signals to new player
    app_controller.window.play_pause_signal.connect(app_controller.toggle_play_pause)
    app_controller.window.frame_changed_signal.connect(app_controller.seek_to_frame)
    new_player.frameChanged.connect(app_controller.update_frame)
    new_player.videoFinished.connect(app_controller.video_finished)
    
    # Integrate the new player's UI components
    integrate_qt_player(app_controller.window, new_player)
    
    return new_player
