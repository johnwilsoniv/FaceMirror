# player_integration.py - Handles integration between the QTMediaPlayer and the MainWindow UI
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLayout


def integrate_qt_player(window, player):
    """
    Integrates the QTMediaPlayer with the MainWindow UI.

    Args:
        window: The MainWindow instance
        player: The QTMediaPlayer instance
    """
    try:
        # Get the video widget from the player
        video_widget = player.get_video_widget()

        # Replace the placeholder video_label with the actual video widget
        if hasattr(window, 'video_label') and window.video_label:
            parent = window.video_label.parent()

            if parent and hasattr(parent, 'layout') and parent.layout():
                parent_layout = parent.layout()

                # Get the index where video_label is in the layout
                index = parent_layout.indexOf(window.video_label)

                # Remove the video_label
                parent_layout.removeWidget(window.video_label)
                window.video_label.setParent(None)
                window.video_label.hide()

                # Add the video widget in its place
                if index >= 0:
                    parent_layout.insertWidget(index, video_widget)
                else:
                    print("Warning: Could not find video_label in layout, adding to end")
                    parent_layout.addWidget(video_widget)
            else:
                print("Warning: No valid parent layout found for video_label")
        else:
            print("Warning: No video_label found in MainWindow")

        # Call the setup method for action display - ADD THIS LINE
        window.setup_action_display()

        # Connect frame updates to action display updates - ADD THIS BLOCK
        player.frameChanged.connect(lambda frame, image, action: window.update_action_display(action))

        # Store a reference to the player in the window for convenience
        window.video_player = player

        print("QT Media Player successfully integrated with UI.")
    except Exception as e:
        print(f"Error integrating video player: {str(e)}")
        import traceback
        traceback.print_exc()