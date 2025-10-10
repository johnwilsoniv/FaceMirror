"""
Progress window for S1 Face Mirror processing.
Displays progress when running as bundled application (no terminal visible).

Version: 3.0 - Scientific High-End Edition
Date: October 9, 2025
Changes: Scientific design, fixed clipping, improved FPS calculation, 8pt grid spacing
"""

import tkinter as tk
from tkinter import ttk, font
from dataclasses import dataclass
from datetime import datetime, timedelta
import queue
import threading
from pathlib import Path
from collections import deque


@dataclass
class ProgressUpdate:
    """Data class for progress updates from worker threads"""
    video_name: str
    video_num: int
    total_videos: int
    stage: str  # 'rotation', 'reading', 'processing', 'writing', 'complete', 'error'
    current: int
    total: int
    message: str = ""
    error: str = ""
    fps: float = 0.0  # Processing rate from tqdm


class ProcessingProgressWindow:
    """
    GUI window that displays processing progress for video conversion.
    Uses queue-based communication to safely receive updates from worker threads.
    Scientific high-end design with precise data visualization.
    """

    def __init__(self, total_videos=1):
        """
        Initialize the progress window.

        Args:
            total_videos: Total number of videos to process
        """
        print(f"[GUI DEBUG] Initializing progress window with {total_videos} total videos")

        self.total_videos = total_videos
        self.current_video = 0
        self.start_time = datetime.now()
        self.stage_start_time = datetime.now()

        # FPS smoothing - rolling average over last 10 updates
        self.fps_history = deque(maxlen=10)
        self.eta_history = deque(maxlen=3)  # Smooth ETA over 3 values
        self.last_update_time = datetime.now()
        self.last_frame_count = 0

        # Create main window
        self.root = tk.Toplevel()
        self.root.title("FaceMirror Processing Pipeline")
        self.root.geometry("760x600")
        self.root.resizable(False, False)

        # Scientific color palette - clinical precision
        self.colors = {
            'primary': '#1a3a52',        # Deep clinical blue
            'primary_light': '#2c5f8d',  # Medium blue
            'accent': '#0066cc',         # Technical blue
            'success': '#00a86b',        # Medical green
            'warning': '#ff9500',        # Amber alert
            'danger': '#d32f2f',         # Clinical red
            'bg': '#f5f7fa',             # Light technical gray
            'bg_card': '#ffffff',        # Pure white
            'bg_alt': '#fafbfc',         # Alternate white
            'border': '#d1d9e0',         # Subtle border
            'border_light': '#e4e9ef',   # Very light border
            'text': '#1a1a1a',           # Near black
            'text_primary': '#2c3e50',   # Dark blue-gray
            'text_secondary': '#546e7a', # Medium gray
            'text_tertiary': '#90a4ae',  # Light gray
            'data_bg': '#f8f9fa',        # Data field background
            'stage_waiting': '#eceff1',  # Stage waiting
            'stage_active': '#0066cc',   # Stage active
            'stage_complete': '#00a86b', # Stage complete
            'grid': '#e8ecef'            # Grid lines
        }

        # Configure window background
        self.root.configure(bg=self.colors['bg'])

        # Make window modal (stay on top)
        self.root.transient()
        self.root.grab_set()

        # Progress update queue (thread-safe)
        self.progress_queue = queue.Queue()

        print(f"[GUI DEBUG] Progress queue created")

        # Setup UI
        self._setup_ui()

        # Start queue monitoring
        self._monitor_queue()

        print(f"[GUI DEBUG] GUI setup complete, monitoring started")

    def _setup_ui(self):
        """Setup the user interface with scientific design"""
        print(f"[GUI DEBUG] Setting up UI...")

        # Configure custom styles
        style = ttk.Style()
        style.theme_use('clam')

        # Scientific progress bar - precise and clean
        style.configure("Scientific.Horizontal.TProgressbar",
                       troughcolor=self.colors['border_light'],
                       background=self.colors['accent'],
                       borderwidth=0,
                       thickness=24)  # Thicker for better visibility

        # Header section - scientific instrument style
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        # Title with technical precision
        title_label = tk.Label(
            header_frame,
            text="FaceMirror Pipeline",
            font=("Helvetica Neue", 18, "normal"),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(pady=(20, 4))

        # Version/subtitle
        version_label = tk.Label(
            header_frame,
            text="Version 1.0",
            font=("Helvetica Neue", 9),
            bg=self.colors['primary'],
            fg='#b0c4de',
            height=2
        )
        version_label.pack()

        # Main container - 8pt grid spacing (24px = 3 units)
        main_frame = tk.Frame(self.root, bg=self.colors['bg'], padx=24, pady=24)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Overall Progress Section
        overall_section = self._create_section_card(main_frame, "OVERALL PROGRESS")
        overall_section.pack(fill=tk.X, pady=(0, 16))

        overall_inner = tk.Frame(overall_section, bg=self.colors['bg_card'], padx=20, pady=16)
        overall_inner.pack(fill=tk.BOTH, expand=True)

        # Video counter with monospaced numbers
        self.overall_label = tk.Label(
            overall_inner,
            text=f"Video 0 of {self.total_videos}",
            font=("SF Mono", 13, "normal") if self._is_mac() else ("Consolas", 13, "normal"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        )
        self.overall_label.pack(anchor=tk.W, pady=(0, 10))

        # Progress bar with technical precision
        self.overall_progress = ttk.Progressbar(
            overall_inner,
            mode='determinate',
            length=700,
            style="Scientific.Horizontal.TProgressbar"
        )
        self.overall_progress.pack(fill=tk.X, pady=(0, 12))

        # Elapsed time - scientific format
        self.elapsed_label = tk.Label(
            overall_inner,
            text="Elapsed: 00:00:00",
            font=("SF Mono", 9) if self._is_mac() else ("Consolas", 9),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        self.elapsed_label.pack(anchor=tk.W)

        # Current Video Section
        video_section = self._create_section_card(main_frame, "CURRENT VIDEO")
        video_section.pack(fill=tk.BOTH, expand=True)

        video_inner = tk.Frame(video_section, bg=self.colors['bg_card'], padx=20, pady=16)
        video_inner.pack(fill=tk.BOTH, expand=True)

        # Video filename with wrapping
        self.video_name_label = tk.Label(
            video_inner,
            text="No video selected",
            font=("Helvetica Neue", 11),
            wraplength=680,
            bg=self.colors['bg_card'],
            fg=self.colors['text'],
            anchor=tk.W,
            justify=tk.LEFT
        )
        self.video_name_label.pack(anchor=tk.W, pady=(0, 16), fill=tk.X)

        # Pipeline stages header
        tk.Label(
            video_inner,
            text="PIPELINE STAGES",
            font=("Helvetica Neue", 9, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary'],
            anchor=tk.W
        ).pack(anchor=tk.W, pady=(0, 8))

        # Stage visualization - scientific grid layout
        stages_frame = tk.Frame(video_inner, bg=self.colors['bg_card'])
        stages_frame.pack(fill=tk.X, pady=(0, 16))

        self.stage_indicators = {}
        stages = [
            ('rotation', 'ROTATION'),
            ('reading', 'READING'),
            ('processing', 'PROCESSING'),
            ('writing', 'WRITING')
        ]

        for i, (stage_key, stage_text) in enumerate(stages):
            stage_container = tk.Frame(stages_frame, bg=self.colors['bg_card'])
            stage_container.pack(side=tk.LEFT, expand=True, fill=tk.X,
                               padx=(0, 6 if i < 3 else 0))

            # Stage indicator box - clean borders
            stage_box = tk.Frame(
                stage_container,
                bg=self.colors['stage_waiting'],
                highlightbackground=self.colors['border'],
                highlightthickness=1
            )
            stage_box.pack(fill=tk.BOTH, expand=True)

            # Stage label
            label = tk.Label(
                stage_box,
                text=stage_text,
                font=("Helvetica Neue", 8, "bold"),
                bg=self.colors['stage_waiting'],
                fg=self.colors['text_tertiary'],
                pady=12,
                padx=10
            )
            label.pack(fill=tk.BOTH, expand=True)
            self.stage_indicators[stage_key] = (stage_box, label)

        # Current stage status
        self.stage_label = tk.Label(
            video_inner,
            text="Status: Initializing",
            font=("Helvetica Neue", 10, "normal"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        )
        self.stage_label.pack(anchor=tk.W, pady=(0, 10))

        # Stage progress bar
        self.stage_progress = ttk.Progressbar(
            video_inner,
            mode='determinate',
            length=700,
            style="Scientific.Horizontal.TProgressbar"
        )
        self.stage_progress.pack(fill=tk.X, pady=(0, 16))

        # Data metrics panel - scientific data presentation
        metrics_container = tk.Frame(
            video_inner,
            bg=self.colors['data_bg'],
            highlightbackground=self.colors['border_light'],
            highlightthickness=1
        )
        metrics_container.pack(fill=tk.X)

        metrics_inner = tk.Frame(metrics_container, bg=self.colors['data_bg'], padx=16, pady=12)
        metrics_inner.pack(fill=tk.BOTH, expand=True)

        # Create a grid for data fields
        data_frame = tk.Frame(metrics_inner, bg=self.colors['data_bg'])
        data_frame.pack(fill=tk.X)

        # Frame count
        self.stage_details_label = tk.Label(
            data_frame,
            text="",
            font=("SF Mono", 9) if self._is_mac() else ("Consolas", 9),
            bg=self.colors['data_bg'],
            fg=self.colors['text'],
            anchor=tk.W
        )
        self.stage_details_label.pack(anchor=tk.W, pady=(0, 6))

        # Performance metrics with wrapping to prevent clipping
        self.stage_eta_label = tk.Label(
            data_frame,
            text="",
            font=("SF Mono", 9) if self._is_mac() else ("Consolas", 9),
            bg=self.colors['data_bg'],
            fg=self.colors['text_secondary'],
            anchor=tk.W,
            wraplength=680,
            justify=tk.LEFT
        )
        self.stage_eta_label.pack(anchor=tk.W)

        # Status bar - clean technical design
        status_bar = tk.Frame(self.root, bg=self.colors['primary'], height=40)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        self.status_label = tk.Label(
            status_bar,
            text="System initializing...",
            font=("Helvetica Neue", 9),
            bg=self.colors['primary'],
            fg='white',
            anchor=tk.W,
            padx=20
        )
        self.status_label.pack(fill=tk.BOTH, expand=True)

        # Update window
        self.root.update()
        print(f"[GUI DEBUG] UI setup complete")

    def _create_section_card(self, parent, title):
        """Create a section card with consistent styling"""
        card = tk.Frame(
            parent,
            bg=self.colors['bg_card'],
            highlightbackground=self.colors['border'],
            highlightthickness=1
        )

        # Section title banner
        title_frame = tk.Frame(card, bg=self.colors['bg_alt'], height=32)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text=title,
            font=("Helvetica Neue", 9, "bold"),
            bg=self.colors['bg_alt'],
            fg=self.colors['text_secondary'],
            anchor=tk.W,
            padx=20
        )
        title_label.pack(fill=tk.BOTH, expand=True)

        return card

    def _is_mac(self):
        """Check if running on macOS"""
        import platform
        return platform.system() == 'Darwin'

    def _monitor_queue(self):
        """Monitor the progress queue and update UI (runs in main thread)"""
        update_count = 0
        try:
            while True:
                update = self.progress_queue.get_nowait()
                update_count += 1
                print(f"[GUI DEBUG] Processing update #{update_count}: stage={update.stage}, video={update.video_num}/{update.total_videos}, progress={update.current}/{update.total}")
                self._apply_update(update)
        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self._monitor_queue)

    def _apply_update(self, update: ProgressUpdate):
        """Apply a progress update to the UI"""
        print(f"[GUI DEBUG] Applying update: {update.video_name}, stage={update.stage}, {update.current}/{update.total}")

        # Update current video tracking
        if update.video_num != self.current_video:
            self.current_video = update.video_num
            self.stage_start_time = datetime.now()
            self.fps_history.clear()  # Reset FPS history for new video
            self.eta_history.clear()  # Reset ETA history for new video
            self.last_frame_count = 0
            print(f"[GUI DEBUG] Video changed to #{update.video_num}")

        # Update overall progress
        overall_pct = ((update.video_num - 1) / self.total_videos) * 100
        if update.stage == 'complete':
            overall_pct = (update.video_num / self.total_videos) * 100

        self.overall_progress['value'] = overall_pct
        self.overall_label.config(
            text=f"Video {update.video_num} of {self.total_videos} — {overall_pct:5.1f}% Complete"
        )

        # Update elapsed time
        elapsed = datetime.now() - self.start_time
        self.elapsed_label.config(
            text=f"Elapsed: {self._format_timedelta(elapsed)}"
        )

        # Update video name
        self.video_name_label.config(text=update.video_name)

        # Update stage indicators
        self._update_stage_indicators(update.stage)

        # Update stage progress
        if update.total > 0:
            stage_pct = (update.current / update.total) * 100
            self.stage_progress['value'] = stage_pct
        else:
            stage_pct = 0
            self.stage_progress['value'] = 0

        # Update stage label
        stage_names = {
            'rotation': 'Video Rotation',
            'reading': 'Frame Acquisition',
            'processing': 'Frame Processing',
            'writing': 'Output Generation',
            'complete': 'Complete',
            'error': 'Error'
        }
        stage_text = stage_names.get(update.stage, update.stage.title())
        self.stage_label.config(text=f"Stage: {stage_text} — {stage_pct:5.1f}%")

        # Update metrics using tqdm's FPS
        if update.stage in ['reading', 'processing', 'writing'] and update.total > 0:
            self.stage_details_label.config(
                text=f"Frames Processed: {update.current:>6,} / {update.total:,}"
            )

            # Use FPS from tqdm if provided, otherwise calculate
            if update.fps > 0:
                print(f"[FPS DEBUG] Using tqdm FPS: {update.fps:.1f}")
                fps_to_display = update.fps

                # Calculate ETA with rolling average for stability
                remaining_frames = update.total - update.current
                eta_seconds = remaining_frames / fps_to_display if fps_to_display > 0 else 0

                # Add to rolling average
                self.eta_history.append(eta_seconds)
                smoothed_eta = sum(self.eta_history) / len(self.eta_history)

                print(f"[ETA DEBUG] Raw ETA: {eta_seconds:.1f}s, Smoothed ETA: {smoothed_eta:.1f}s")

                self.stage_eta_label.config(
                    text=f"Processing Rate: {fps_to_display:>6.1f} fps    |    Est. Remaining: {self._format_seconds(smoothed_eta)}"
                )
            elif update.current > 0:
                # Fallback: only show if we have processed some frames
                self.stage_eta_label.config(text="Calculating performance metrics...")
            else:
                self.stage_eta_label.config(text="Initializing...")

        elif update.stage == 'rotation':
            self.stage_details_label.config(text="Analyzing and correcting video orientation...")
            self.stage_eta_label.config(text="")
        else:
            self.stage_details_label.config(text="")
            self.stage_eta_label.config(text="")

        # Update status bar
        if update.error:
            self.status_label.config(
                text=f"ERROR: {update.error}",
                bg=self.colors['danger']
            )
        elif update.message:
            self.status_label.config(
                text=update.message,
                bg=self.colors['primary']
            )
        elif update.stage == 'complete':
            self.status_label.config(
                text="Processing complete — All operations successful",
                bg=self.colors['success']
            )

        # Force UI update
        self.root.update()
        print(f"[GUI DEBUG] UI updated successfully")

    def _update_stage_indicators(self, current_stage):
        """Update the visual pipeline stage indicators"""
        stage_order = ['rotation', 'reading', 'processing', 'writing']

        print(f"[GUI DEBUG] Updating stage indicators, current stage: {current_stage}")

        # Reset all to default
        for stage_key, (box, label) in self.stage_indicators.items():
            box.config(bg=self.colors['stage_waiting'])
            label.config(
                bg=self.colors['stage_waiting'],
                fg=self.colors['text_tertiary'],
                font=("Helvetica Neue", 8, "bold")
            )

        # Update based on current stage
        try:
            current_idx = stage_order.index(current_stage)

            # Mark completed stages
            for i in range(current_idx):
                stage = stage_order[i]
                if stage in self.stage_indicators:
                    box, label = self.stage_indicators[stage]
                    box.config(bg=self.colors['stage_complete'])
                    label.config(
                        bg=self.colors['stage_complete'],
                        fg='white',
                        font=("Helvetica Neue", 8, "bold")
                    )

            # Mark active stage
            if current_stage in self.stage_indicators:
                box, label = self.stage_indicators[current_stage]
                box.config(bg=self.colors['stage_active'])
                label.config(
                    bg=self.colors['stage_active'],
                    fg='white',
                    font=("Helvetica Neue", 9, "bold")
                )

        except ValueError:
            # Handle complete/error
            if current_stage == 'complete':
                for box, label in self.stage_indicators.values():
                    box.config(bg=self.colors['stage_complete'])
                    label.config(
                        bg=self.colors['stage_complete'],
                        fg='white',
                        font=("Helvetica Neue", 8, "bold")
                    )

    def update_progress(self, update: ProgressUpdate):
        """
        Thread-safe method to send progress updates from worker threads.

        Args:
            update: ProgressUpdate object with current progress
        """
        print(f"[GUI DEBUG] update_progress called: stage={update.stage}, video={update.video_num}/{update.total_videos}")
        self.progress_queue.put(update)
        print(f"[GUI DEBUG] Update added to queue (queue size: {self.progress_queue.qsize()})")

        # CRITICAL FIX: Force the event loop to run so _monitor_queue can process the queue
        # This is necessary because video processing blocks the main thread
        try:
            self.root.update()
            print(f"[GUI DEBUG] Forced GUI update after adding to queue")
        except Exception as e:
            print(f"[GUI DEBUG] Error forcing GUI update: {e}")

    def close(self):
        """Close the progress window"""
        print(f"[GUI DEBUG] Closing progress window")
        if self.root:
            try:
                self.root.destroy()
            except:
                pass

    @staticmethod
    def _format_timedelta(td):
        """Format timedelta as HH:MM:SS"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _format_seconds(seconds):
        """Format seconds as MM:SS or HH:MM:SS"""
        seconds = int(seconds)
        if seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:02d}:{secs:02d}"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"


# Test code
if __name__ == "__main__":
    import time

    def test_progress_window():
        """Test the progress window with simulated updates"""
        root = tk.Tk()
        root.withdraw()

        print("[TEST] Creating progress window...")
        window = ProcessingProgressWindow(total_videos=2)

        # Simulate processing 2 videos
        for video_num in range(1, 3):
            video_name = f"patient_facial_exam_{video_num:02d}_session_20250109.mp4"

            # Rotation stage
            print(f"[TEST] Sending rotation updates for video {video_num}")
            for i in range(51):
                window.update_progress(ProgressUpdate(
                    video_name=video_name,
                    video_num=video_num,
                    total_videos=2,
                    stage='rotation',
                    current=i * 2,
                    total=100,
                    message="Checking video rotation..."
                ))
                time.sleep(0.02)

            # Reading stage
            print(f"[TEST] Sending reading updates for video {video_num}")
            for i in range(101):
                window.update_progress(ProgressUpdate(
                    video_name=video_name,
                    video_num=video_num,
                    total_videos=2,
                    stage='reading',
                    current=i * 10,
                    total=1000,
                    message="Reading frames into memory..."
                ))
                time.sleep(0.02)

            # Processing stage
            print(f"[TEST] Sending processing updates for video {video_num}")
            for i in range(101):
                window.update_progress(ProgressUpdate(
                    video_name=video_name,
                    video_num=video_num,
                    total_videos=2,
                    stage='processing',
                    current=i * 10,
                    total=1000,
                    message="Processing frames with face detection...",
                    fps=45.3  # Simulated FPS
                ))
                time.sleep(0.02)

            # Writing stage
            print(f"[TEST] Sending writing updates for video {video_num}")
            for i in range(101):
                window.update_progress(ProgressUpdate(
                    video_name=video_name,
                    video_num=video_num,
                    total_videos=2,
                    stage='writing',
                    current=i * 10,
                    total=1000,
                    message="Writing output files..."
                ))
                time.sleep(0.02)

            # Complete
            print(f"[TEST] Sending complete for video {video_num}")
            window.update_progress(ProgressUpdate(
                video_name=video_name,
                video_num=video_num,
                total_videos=2,
                stage='complete',
                current=1000,
                total=1000,
                message="Video processing complete"
            ))

        time.sleep(2)
        window.close()
        root.destroy()

    test_progress_window()
