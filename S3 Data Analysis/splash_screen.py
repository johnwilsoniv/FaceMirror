"""
Splash screen module for SplitFace applications
Provides visual feedback during application startup
"""

import tkinter as tk
from tkinter import ttk
import threading


class SplashScreen:
    """Lightweight splash screen shown during app initialization"""

    def __init__(self, app_name, version="2.0.0"):
        self.app_name = app_name
        self.version = version
        self.window = None
        self.status_label = None
        self.progress_bar = None
        self._should_close = False

    def show(self):
        """Display the splash screen"""
        self.window = tk.Tk()
        self.window.title("")
        self.window.overrideredirect(True)  # Remove window decorations

        # Calculate center position
        window_width = 400
        window_height = 250
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')

        # Create main frame with border
        main_frame = tk.Frame(
            self.window,
            bg='white',
            relief='solid',
            borderwidth=1
        )
        main_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # App name (large, bold)
        name_label = tk.Label(
            main_frame,
            text=self.app_name,
            font=('Helvetica', 28, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        name_label.pack(pady=(40, 5))

        # Version
        version_label = tk.Label(
            main_frame,
            text=f"Version {self.version}",
            font=('Helvetica', 11),
            bg='white',
            fg='#7f8c8d'
        )
        version_label.pack(pady=(0, 30))

        # Status message
        self.status_label = tk.Label(
            main_frame,
            text="Initializing...",
            font=('Helvetica', 12),
            bg='white',
            fg='#34495e'
        )
        self.status_label.pack(pady=(0, 15))

        # Progress bar (indeterminate mode)
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Splash.Horizontal.TProgressbar",
                       troughcolor='#ecf0f1',
                       background='#3498db',
                       thickness=8)

        self.progress_bar = ttk.Progressbar(
            main_frame,
            style="Splash.Horizontal.TProgressbar",
            mode='indeterminate',
            length=300
        )
        self.progress_bar.pack(pady=(0, 40))
        self.progress_bar.start(10)  # Animation speed

        # Keep window on top
        self.window.attributes('-topmost', True)

        # Update the window
        self.window.update()

    def update_status(self, message):
        """Update the status message"""
        if self.window and self.status_label:
            self.status_label.config(text=message)
            self.window.update()

    def close(self):
        """Close the splash screen"""
        if self.window:
            try:
                self.progress_bar.stop()
                self.window.destroy()
            except:
                pass
            self.window = None

    def get_root_and_close(self):
        """
        Close splash screen but return the root Tk instance for reuse.
        This avoids issues with creating multiple Tk instances on macOS.
        """
        root = self.window
        if root:
            try:
                self.progress_bar.stop()
                # Clear all widgets from the window instead of destroying it
                for widget in root.winfo_children():
                    widget.destroy()
                # Reset window properties
                root.overrideredirect(False)
                root.attributes('-topmost', False)
            except:
                pass
            self.window = None
        return root

    def __enter__(self):
        """Context manager support"""
        self.show()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()
        return False


def show_splash_during_import(app_name, stages, import_func):
    """
    Show splash screen with stages while performing imports

    Args:
        app_name: Name of the application
        stages: List of (message, callable) tuples
        import_func: Function to call after showing splash

    Returns:
        Result of import_func()
    """
    splash = SplashScreen(app_name)
    splash.show()

    try:
        for message, func in stages:
            splash.update_status(message)
            if func:
                func()

        # Final stage
        splash.update_status("Starting application...")
        result = import_func() if import_func else None

        splash.close()
        return result

    except Exception as e:
        splash.close()
        raise
