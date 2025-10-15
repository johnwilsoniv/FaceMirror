"""
Mode Selection Window for S1 Face Mirror
Provides three processing workflows:
1. Mirror + OpenFace (complete pipeline)
2. Mirror Only (no AU extraction)
3. OpenFace Only (process existing mirrored videos)
"""

import tkinter as tk
from tkinter import ttk
import torch


class ModeSelectionWindow:
    """Main window for selecting processing mode"""

    def __init__(self, on_mode_selected):
        """
        Initialize mode selection window

        Args:
            on_mode_selected: Callback function(mode) where mode is 'mirror_openface', 'mirror_only', or 'openface_only'
        """
        self.on_mode_selected = on_mode_selected

        # Detect GPU
        self.device = self._detect_device()

        # Create window
        self.root = tk.Tk()
        self.root.title("S1 Face Mirror — Processing Mode Selection")
        self.root.geometry("900x550")
        self.root.resizable(False, False)

        # Scientific color palette (matching progress_window.py)
        self.colors = {
            'primary': '#1a3a52',
            'accent': '#0066cc',
            'success': '#00a86b',
            'bg': '#f5f7fa',
            'bg_card': '#ffffff',
            'border': '#d1d9e0',
            'text': '#1a1a1a',
            'text_primary': '#2c3e50',
            'text_secondary': '#546e7a',
        }

        self.root.configure(bg=self.colors['bg'])

        self._setup_ui()

    def _detect_device(self):
        """
        Auto-detect best available device

        Note: MPS is not supported by OpenFace 3.0 face detection models.
        Falls back to CPU on Apple Silicon.
        """
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {device_name}")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Apple MPS detected, but not supported by face detection models")
            print("Falling back to CPU for compatibility")
            return 'cpu'
        else:
            print("Using CPU (no GPU detected)")
            return 'cpu'

    def _setup_ui(self):
        """Setup the user interface"""

        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="Face Mirror Pipeline",
            font=("Helvetica Neue", 18, "normal"),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(pady=(25, 5))

        device_label = tk.Label(
            header_frame,
            text=f"Version 1.0  |  Device: {self.device.upper()}",
            font=("Helvetica Neue", 10),
            bg=self.colors['primary'],
            fg='#b0c4de'
        )
        device_label.pack()

        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['bg'], padx=40, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Section title
        section_card = self._create_section_card(main_frame, "SELECT PROCESSING WORKFLOW")
        section_card.pack(fill=tk.X, pady=(0, 20))

        # Button container
        button_container = tk.Frame(main_frame, bg=self.colors['bg'])
        button_container.pack(fill=tk.BOTH, expand=True)

        # Create three mode buttons
        self._create_mode_button(
            button_container,
            icon="1",
            title="FACE MIRROR + OPENFACE",
            description="Process new videos through\nthe complete pipeline",
            color=self.colors['success'],
            mode='mirror_openface',
            column=0
        )

        self._create_mode_button(
            button_container,
            icon="2",
            title="FACE MIRROR ONLY",
            description="Process new videos\nwithout AU analysis",
            color=self.colors['accent'],
            mode='mirror_only',
            column=1
        )

        self._create_mode_button(
            button_container,
            icon="3",
            title="OPENFACE ONLY",
            description="Process previously mirrored\nvideos through OpenFace",
            color='#ff9500',
            mode='openface_only',
            column=2
        )

    def _create_section_card(self, parent, title):
        """Create a section card with consistent styling"""
        card = tk.Frame(
            parent,
            bg=self.colors['bg_card'],
            highlightbackground=self.colors['border'],
            highlightthickness=1
        )

        title_frame = tk.Frame(card, bg='#fafbfc', height=40)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text=title,
            font=("Helvetica Neue", 10, "bold"),
            bg='#fafbfc',
            fg=self.colors['text_secondary'],
            anchor=tk.W,
            padx=20
        )
        title_label.pack(fill=tk.BOTH, expand=True)

        return card

    def _create_mode_button(self, parent, icon, title, description, color, mode, column):
        """Create a mode selection button"""
        button_frame = tk.Frame(
            parent,
            bg=self.colors['bg_card'],
            highlightbackground=self.colors['border'],
            highlightthickness=2
        )
        button_frame.grid(row=0, column=column, padx=10, sticky='nsew')
        parent.grid_columnconfigure(column, weight=1, uniform='button')

        # Single click handler with immediate visual feedback
        def on_click(e=None):
            # Immediate visual feedback - flash the selection
            button_frame.config(highlightbackground=color, highlightthickness=4, bg='#f0f0f0')
            button_frame.update()
            self.root.destroy()
            self.on_mode_selected(mode)

        # Hover effects with cursor change
        def on_enter(e):
            button_frame.config(highlightbackground=color, highlightthickness=3)

        def on_leave(e):
            button_frame.config(highlightbackground=self.colors['border'], highlightthickness=2)

        # Icon (mode number)
        icon_label = tk.Label(
            button_frame,
            text=icon,
            font=("Helvetica Neue", 72, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary'],
            cursor='hand2'
        )
        icon_label.pack(pady=(20, 10))

        # Title
        title_label = tk.Label(
            button_frame,
            text=title,
            font=("Helvetica Neue", 14, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            cursor='hand2'
        )
        title_label.pack(pady=(0, 10))

        # Description
        desc_label = tk.Label(
            button_frame,
            text=description,
            font=("Helvetica Neue", 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary'],
            justify=tk.CENTER,
            cursor='hand2'
        )
        desc_label.pack(pady=(0, 30), padx=20)

        # Set cursor for button frame
        button_frame.config(cursor='hand2')

        # Recursively bind click event to button_frame and all children
        def bind_click_recursively(widget):
            widget.bind('<Button-1>', on_click)
            # Also set hand cursor for all widgets
            try:
                widget.config(cursor='hand2')
            except:
                pass
            for child in widget.winfo_children():
                bind_click_recursively(child)

        # Recursively bind hover effects to frame and all children
        def bind_hover_recursively(widget):
            widget.bind('<Enter>', on_enter)
            widget.bind('<Leave>', on_leave)
            for child in widget.winfo_children():
                bind_hover_recursively(child)

        # Bind all events
        bind_click_recursively(button_frame)
        bind_hover_recursively(button_frame)

    def run(self):
        """Run the window"""
        self.root.mainloop()


def show_mode_selection(on_mode_selected):
    """
    Show mode selection window and return selected mode

    Args:
        on_mode_selected: Callback function(mode)
    """
    window = ModeSelectionWindow(on_mode_selected)
    window.run()


if __name__ == "__main__":
    # Test
    def test_callback(mode):
        print(f"Selected mode: {mode}")

    show_mode_selection(test_callback)
