# --- START OF FILE player_integration.py ---

# player_integration.py - Handles integration between the QTMediaPlayer and the MainWindow UI
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLayout, QLabel, QSizePolicy # Added QLabel, QSizePolicy

def integrate_qt_player(window, player):
    """
    Integrates the QTMediaPlayer with the MainWindow UI.
    Finds the placeholder widget by object name and replaces it.

    Args:
        window: The MainWindow instance
        player: The QTMediaPlayer instance
    """
    try:
        # Get the actual video widget container from the player
        player_container_widget = player.get_video_widget() # This is the QWidget holding the QVideoWidget

        # Find the placeholder widget in the MainWindow using its object name
        placeholder_widget = window.findChild(QLabel, "videoPlaceholder") # Find by name

        if placeholder_widget:
            # Get the parent layout of the placeholder
            parent_layout = placeholder_widget.parentWidget().layout()

            if parent_layout:
                # Replace the placeholder with the player's widget container
                index = parent_layout.indexOf(placeholder_widget)
                if index != -1:
                    # Remove placeholder
                    parent_layout.removeWidget(placeholder_widget)
                    placeholder_widget.setParent(None)
                    placeholder_widget.deleteLater() # Ensure it's properly deleted

                    # Ensure the player widget can expand
                    player_container_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                    # Add player widget container
                    parent_layout.insertWidget(index, player_container_widget, 1) # Add with stretch factor 1
                    print("QT Media Player view integrated into main layout via findChild.")

                    # Now that the real widget is added, update the window's reference
                    # This allows other parts of the code (like update_video_frame) to potentially
                    # interact with the container if needed, although direct interaction is minimal now.
                    window.video_display_widget = player_container_widget

                else:
                    print("CRITICAL ERROR: Could not find placeholder_widget index in parent layout.")
            else:
                print("CRITICAL ERROR: Parent layout not found for placeholder_widget.")
        else:
            print("CRITICAL ERROR: Could not find QLabel with objectName='videoPlaceholder'. Integration failed.")

    except Exception as e:
        print(f"Error integrating video player: {str(e)}")
        import traceback
        traceback.print_exc()
# --- END OF FILE player_integration.py ---