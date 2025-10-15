"""
Patient Browser Window for OpenFace-Only Mode
Scans mirrored videos folder and shows processing status
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import List


class PatientBrowserWindow:
    """Browser for selecting existing mirrored videos to process with OpenFace"""

    def __init__(self, mirrored_videos_dir, openface_output_dir):
        """
        Initialize patient browser

        Args:
            mirrored_videos_dir: Path to folder containing mirrored videos
            openface_output_dir: Path to folder containing OpenFace CSV outputs
        """
        self.mirrored_videos_dir = Path(mirrored_videos_dir)
        self.openface_output_dir = Path(openface_output_dir)
        self.selected_files = []

        # Create window
        self.root = tk.Toplevel()
        self.root.title("OpenFace Only — Select Videos to Process")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # Scientific color palette
        self.colors = {
            'primary': '#1a3a52',
            'accent': '#0066cc',
            'success': '#00a86b',
            'warning': '#ff9500',
            'bg': '#f5f7fa',
            'bg_card': '#ffffff',
            'border': '#d1d9e0',
            'text': '#1a1a1a',
            'text_primary': '#2c3e50',
            'text_secondary': '#546e7a',
        }

        self.root.configure(bg=self.colors['bg'])

        # Scan videos
        self.videos = self._scan_videos()

        # Filter state
        self.filter_mode = tk.StringVar(value='all')
        self.search_text = tk.StringVar()
        self.search_text.trace('w', lambda *args: self._apply_filters())

        self._setup_ui()
        self._populate_tree()

    def _scan_videos(self):
        """Scan mirrored videos folder and check processing status"""
        videos = []

        if not self.mirrored_videos_dir.exists():
            print(f"Warning: Mirrored videos directory not found: {self.mirrored_videos_dir}")
            return videos

        # Find all mirrored video files
        for video_file in self.mirrored_videos_dir.glob('*_mirrored.*'):
            if video_file.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
                # Skip debug videos
                if 'debug' in video_file.name.lower():
                    continue

                # Check if CSV exists
                csv_path = self.openface_output_dir / f"{video_file.stem}.csv"
                status = 'processed' if csv_path.exists() else 'pending'

                videos.append({
                    'path': video_file,
                    'name': video_file.name,
                    'status': status,
                    'csv_path': csv_path
                })

        # Sort by name
        videos.sort(key=lambda x: x['name'])
        return videos

    def _setup_ui(self):
        """Setup the user interface"""

        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="Select Videos for OpenFace Processing",
            font=("Helvetica Neue", 16, "normal"),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(pady=(20, 5))

        # Count processed and pending
        processed_count = sum(1 for v in self.videos if v['status'] == 'processed')
        pending_count = len(self.videos) - processed_count

        source_label = tk.Label(
            header_frame,
            text=f"Source: {self.mirrored_videos_dir.name}/  |  Found: {len(self.videos)} videos  |  Processed: {processed_count}  |  Pending: {pending_count}",
            font=("Helvetica Neue", 9),
            bg=self.colors['primary'],
            fg='#b0c4de'
        )
        source_label.pack()

        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['bg'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Controls frame
        controls_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        controls_frame.pack(fill=tk.X, pady=(0, 15))

        # Search box
        tk.Label(
            controls_frame,
            text="Search:",
            font=("Helvetica Neue", 10),
            bg=self.colors['bg'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT, padx=(0, 10))

        search_entry = tk.Entry(
            controls_frame,
            textvariable=self.search_text,
            font=("Helvetica Neue", 10),
            width=30
        )
        search_entry.pack(side=tk.LEFT, padx=(0, 20))

        # Refresh button
        tk.Button(
            controls_frame,
            text="Refresh",
            command=self._refresh,
            font=("Helvetica Neue", 9),
            bg=self.colors['bg_card'],
            relief=tk.RAISED,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT)

        # Filter frame
        filter_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        filter_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            filter_frame,
            text="Status Filter:",
            font=("Helvetica Neue", 10),
            bg=self.colors['bg'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT, padx=(0, 10))

        for value, text in [('all', 'All'), ('processed', '✓ Processed'), ('pending', '⚠ Unprocessed')]:
            tk.Radiobutton(
                filter_frame,
                text=text,
                variable=self.filter_mode,
                value=value,
                command=self._apply_filters,
                font=("Helvetica Neue", 9),
                bg=self.colors['bg']
            ).pack(side=tk.LEFT, padx=5)

        # Tree frame with scrollbar
        tree_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview
        columns = ('filename', 'status')
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show='tree headings',
            selectmode='none',
            height=15
        )

        # Define columns
        self.tree.heading('#0', text='☑')
        self.tree.heading('filename', text='File Name')
        self.tree.heading('status', text='Status')

        self.tree.column('#0', width=50, anchor=tk.CENTER)
        self.tree.column('filename', width=500, anchor=tk.W)
        self.tree.column('status', width=150, anchor=tk.CENTER)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind click to toggle checkbox
        self.tree.bind('<Button-1>', self._on_tree_click)

        # Selection controls
        selection_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        selection_frame.pack(fill=tk.X, pady=(15, 0))

        tk.Button(
            selection_frame,
            text="☑ Select All",
            command=self._select_all,
            font=("Helvetica Neue", 9),
            bg=self.colors['bg_card'],
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            selection_frame,
            text="☐ Deselect All",
            command=self._deselect_all,
            font=("Helvetica Neue", 9),
            bg=self.colors['bg_card'],
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            selection_frame,
            text="⚠ Select Unprocessed Only",
            command=self._select_unprocessed,
            font=("Helvetica Neue", 9),
            bg=self.colors['bg_card'],
            padx=10,
            pady=5
        ).pack(side=tk.LEFT)

        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text="",
            font=("Helvetica Neue", 10, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text_primary']
        )
        self.status_label.pack(fill=tk.X, pady=(15, 0))

        # Button frame
        button_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        button_frame.pack(fill=tk.X, pady=(15, 0))

        tk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel,
            font=("Helvetica Neue", 11),
            bg='#e0e0e0',
            fg=self.colors['text'],
            padx=20,
            pady=10,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            button_frame,
            text="▶ Start Processing",
            command=self._start_processing,
            font=("Helvetica Neue", 11, "bold"),
            bg=self.colors['success'],
            fg='white',
            padx=20,
            pady=10,
            relief=tk.RAISED
        ).pack(side=tk.LEFT)

    def _populate_tree(self):
        """Populate tree with video files"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Add videos
        for video in self.videos:
            icon = '✓' if video['status'] == 'processed' else '⚠'
            status_text = f"{icon} {'Processed' if video['status'] == 'processed' else 'Pending'}"

            # Add with checkbox placeholder
            self.tree.insert(
                '',
                'end',
                text='☐',
                values=(video['name'], status_text),
                tags=(video['status'],)
            )

        # Configure tags for colors with backgrounds
        self.tree.tag_configure('processed', foreground='#00a86b', background='#e8f5e9')
        self.tree.tag_configure('pending', foreground='#ff9500', background='#fff3e0')

        self._update_status()

    def _apply_filters(self):
        """Apply search and filter to tree"""
        filter_mode = self.filter_mode.get()
        search = self.search_text.get().lower()

        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Re-populate with filtered items
        for video in self.videos:
            # Apply status filter
            if filter_mode != 'all' and video['status'] != filter_mode:
                continue

            # Apply search filter
            if search and search not in video['name'].lower():
                continue

            icon = '✓' if video['status'] == 'processed' else '⚠'
            status_text = f"{icon} {'Processed' if video['status'] == 'processed' else 'Pending'}"

            self.tree.insert(
                '',
                'end',
                text='☐',
                values=(video['name'], status_text),
                tags=(video['status'],)
            )

        # Configure tags for colors with backgrounds
        self.tree.tag_configure('processed', foreground='#00a86b', background='#e8f5e9')
        self.tree.tag_configure('pending', foreground='#ff9500', background='#fff3e0')

        self._update_status()

    def _on_tree_click(self, event):
        """Handle tree click to toggle checkbox"""
        region = self.tree.identify_region(event.x, event.y)
        if region == 'tree':
            item = self.tree.identify_row(event.y)
            if item:
                # Toggle checkbox
                current = self.tree.item(item, 'text')
                new_state = '☐' if current == '☑' else '☑'
                self.tree.item(item, text=new_state)
                self._update_status()

    def _select_all(self):
        """Select all visible items"""
        for item in self.tree.get_children():
            self.tree.item(item, text='☑')
        self._update_status()

    def _deselect_all(self):
        """Deselect all items"""
        for item in self.tree.get_children():
            self.tree.item(item, text='☐')
        self._update_status()

    def _select_unprocessed(self):
        """Select only unprocessed items"""
        for item in self.tree.get_children():
            tags = self.tree.item(item, 'tags')
            if 'pending' in tags:
                self.tree.item(item, text='☑')
            else:
                self.tree.item(item, text='☐')
        self._update_status()

    def _update_status(self):
        """Update status label with selection count"""
        selected_count = sum(1 for item in self.tree.get_children() if self.tree.item(item, 'text') == '☑')

        # Estimate time (rough: 1 minute per 1000 frames, assuming ~30fps and 30 second videos = ~900 frames)
        # Very rough estimate: 1 minute per video
        estimated_minutes = selected_count * 1

        self.status_label.config(
            text=f"Selected: {selected_count} videos | Estimated Time: ~{estimated_minutes} minute{'s' if estimated_minutes != 1 else ''}"
        )

    def _refresh(self):
        """Refresh video list"""
        self.videos = self._scan_videos()
        self._apply_filters()

    def _cancel(self):
        """Cancel and close window"""
        self.selected_files = []
        self.root.destroy()

    def _start_processing(self):
        """Start processing selected files"""
        # Get selected files
        self.selected_files = []
        for item in self.tree.get_children():
            if self.tree.item(item, 'text') == '☑':
                filename = self.tree.item(item, 'values')[0]
                # Find the video dict
                video = next((v for v in self.videos if v['name'] == filename), None)
                if video:
                    self.selected_files.append(str(video['path']))

        if not self.selected_files:
            messagebox.showwarning("No Selection", "Please select at least one video to process.")
            return

        # Close window
        self.root.destroy()

    def show(self) -> List[str]:
        """
        Show browser and return selected files

        Returns:
            List of selected file paths
        """
        self.root.wait_window()
        return self.selected_files


def browse_patients(mirrored_videos_dir, openface_output_dir) -> List[str]:
    """
    Show patient browser and return selected files

    Args:
        mirrored_videos_dir: Path to mirrored videos folder
        openface_output_dir: Path to OpenFace output folder

    Returns:
        List of selected file paths
    """
    browser = PatientBrowserWindow(mirrored_videos_dir, openface_output_dir)
    return browser.show()


if __name__ == "__main__":
    # Test
    from pathlib import Path

    test_dir = Path.cwd().parent / 'S1O Processed Files' / 'Face Mirror 1.0 Output'
    test_output = Path.cwd().parent / 'S1O Processed Files' / 'Combined Data'

    selected = browse_patients(test_dir, test_output)
    print(f"Selected {len(selected)} files:")
    for f in selected:
        print(f"  - {f}")
