class ProgressWindow:
    def __init__(self, title="Processing Videos", command_queue=None):
        self.root = tk.Tk()
        self.root.title(title)
        self.command_queue = command_queue

        # Setup periodic queue check
        self.check_queue()

        # Set window size and position
        window_width, window_height = 400, 150
        x = (self.root.winfo_screenwidth() - window_width) // 2
        y = (self.root.winfo_screenheight() - window_height) // 2
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')

        # Create UI elements
        self.setup_ui()

        # Initialize progress variables
        self.total_frames = 0
        self.current_frame = 0
        self.current_file = ""
        self.file_number = 0
        self.total_files = 0

        # Configure window properties
        self.root.transient()
        self.root.lift()
        self.root.resizable(False, False)

    def setup_ui(self):
        """Create and layout UI elements"""
        self.label = ttk.Label(self.root, text="Initializing...", padding=(10, 5))
        self.progress = ttk.Progressbar(self.root, length=300, mode='determinate')
        self.status = ttk.Label(self.root, text="", padding=(10, 5))

        self.label.pack()
        self.progress.pack(pady=10)
        self.status.pack()

    def set_total_files(self, total):
        self.total_files = total
        self.file_number = 0

    def update_file_progress(self, filename, total_frames):
        self.current_file = filename
        self.total_frames = total_frames
        self.current_frame = 0
        self.file_number += 1
        self.update_display()

    def increment_progress(self):
        self.current_frame += 1
        if self.current_frame % 30 == 0:  # Reduced update frequency
            self.update_display()

    def update_display(self):
        if self.total_frames > 0:
            progress = (self.current_frame / self.total_frames) * 100
            self.progress['value'] = progress

            self.label['text'] = f"{Path(self.current_file).name}"
            self.status['text'] = (f"Processing file {self.file_number} of {self.total_files}\n"
                                   f"Frame {self.current_frame} of {self.total_frames} ({progress:.1f}%)")

        self.root.update()

    def check_queue(self):
        """Check for commands in the queue"""
        try:
            if self.command_queue:
                while True:
                    try:
                        command = self.command_queue.get_nowait()
                        if command == "close":
                            self.root.quit()
                            return
                    except queue.Empty:
                        break
        except tk.TclError:
            return

        # Schedule next check
        self.root.after(100, self.check_queue)

    def close(self):
        """Schedule window closure through the event loop"""
        if self.command_queue:
            self.command_queue.put("close")
        else:
            self.root.quit()
