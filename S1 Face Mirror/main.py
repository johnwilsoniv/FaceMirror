import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
from pathlib import Path
from face_splitter import StableFaceSplitter
from openface_integration import run_openface_processing
from multiprocessing import cpu_count

class OpenFaceOptionsDialog(simpledialog.Dialog):
    """
    A dialog that asks the user to choose from three OpenFace processing options.
    Uses tkinter's simpledialog.Dialog for reliable modal behavior.
    """
    def __init__(self, parent, title, message):
        self.message = message
        self.result = "none"  # Default result
        super().__init__(parent, title)
    
    def body(self, master):
        """Create dialog body. Called by __init__."""
        tk.Label(master, text=self.message, wraplength=400, justify=tk.LEFT).grid(
            row=0, column=0, padx=20, pady=20, sticky=tk.W+tk.E)
        return None  # No initial focus
    
    def buttonbox(self):
        """Add custom buttons. Overrides the standard buttons."""
        box = tk.Frame(self)
        
        w = tk.Button(box, text="Run all files", width=16, command=lambda: self.set_result("all"), 
                     default=tk.ACTIVE)
        w.pack(side=tk.LEFT, padx=5, pady=10)
        
        w = tk.Button(box, text="Run session files only", width=16, command=lambda: self.set_result("session"))
        w.pack(side=tk.LEFT, padx=5, pady=10)
        
        w = tk.Button(box, text="Do not run", width=16, command=lambda: self.set_result("none"))
        w.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.bind("<Return>", lambda event: self.set_result("all"))
        self.bind("<Escape>", lambda event: self.set_result("none"))
        
        box.pack()
    
    def set_result(self, value):
        """Set the result and close the dialog"""
        self.result = value
        self.ok()  # This will destroy the dialog

def process_single_video(args):
    """
    Worker function for parallel video processing.
    Must be a top-level function for multiprocessing to work.

    Args:
        args: tuple of (input_path, output_dir, debug_mode)

    Returns:
        dict with processing results
    """
    input_path, output_dir, debug_mode = args
    try:
        # Create a new splitter instance for this worker
        # Each worker gets its own detector to avoid sharing state
        splitter = StableFaceSplitter(debug_mode=debug_mode)
        outputs = splitter.process_video(input_path, output_dir)
        return {
            'input': input_path,
            'success': True,
            'outputs': outputs
        }
    except Exception as e:
        return {
            'input': input_path,
            'success': False,
            'error': str(e)
        }

def main():
    """Simple command-line interface"""
    try:
        root = tk.Tk()
        root.withdraw()
        input_paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if not input_paths:
            root.destroy()
            return

        output_dir = Path.cwd() / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        debug_mode = True

        print(f"\nProcessing {len(input_paths)} video(s) sequentially...")
        print(f"Available CPU cores: {cpu_count()}")
        print(f"Each video will use multi-threaded frame processing\n")

        # Process videos sequentially (one at a time)
        # Each video uses multi-threading internally for optimal performance
        results = []
        for video_num, input_path in enumerate(input_paths, 1):
            print(f"\n{'='*60}")
            print(f"VIDEO {video_num} of {len(input_paths)}: {Path(input_path).name}")
            print(f"{'='*60}")
            result = process_single_video((input_path, output_dir, debug_mode))
            results.append(result)
            print(f"\nCompleted {video_num}/{len(input_paths)} videos")

        summary = "Processing Results:\n\n"
        for result in results:
            if result['success']:
                summary += f"✓ {Path(result['input']).name}\n"

                # Separate output types
                left_video = next((f for f in result['outputs'] if 'left_mirrored' in f), None)
                right_video = next((f for f in result['outputs'] if 'right_mirrored' in f), None)
                debug_video = next((f for f in result['outputs'] if 'debug' in f), None)
                rotated_video = next((f for f in result['outputs'] if 'rotated' in f), None)

                if left_video:
                    summary += f"  - Left: {Path(left_video).name}\n"
                if right_video:
                    summary += f"  - Right: {Path(right_video).name}\n"
                if debug_video:
                    summary += f"  - Debug: {Path(debug_video).name}\n"
                if rotated_video:
                    summary += f"  - Rotated: {Path(rotated_video).name}\n"
            else:
                summary += f"✗ {Path(result['input']).name} - Error: {result['error']}\n"

        messagebox.showinfo("Processing Complete", summary)
        
        # Extract all mirrored videos from the current session's results
        session_mirrored_files = []
        for result in results:
            if result['success']:
                for output in result['outputs']:
                    if 'mirrored' in output and 'debug' not in output:
                        session_mirrored_files.append(output)
        
        # Show dialog with OpenFace processing options using simpledialog.Dialog
        openface_message = "Would you like to run OpenFace processing on the mirrored videos?\n\nThis will process videos with OpenFace to extract facial features."
        dialog = OpenFaceOptionsDialog(
            root, 
            "OpenFace Processing", 
            openface_message
        )
        openface_choice = dialog.result
        
        if openface_choice == "all":
            try:
                # Process all files in the output directory
                processed_count = run_openface_processing(output_dir, move_files=True)
                
                if processed_count > 0:
                    messagebox.showinfo(
                        "OpenFace Processing Complete",
                        f"{processed_count} files were processed with OpenFace and moved to the 'S1O Processed Files' directory."
                    )
                else:
                    messagebox.showinfo(
                        "OpenFace Processing",
                        "No files were processed with OpenFace."
                    )
            except Exception as e:
                messagebox.showerror(
                    "OpenFace Processing Error",
                    f"Error during OpenFace processing: {str(e)}"
                )
        elif openface_choice == "session":
            try:
                # Process only files from this session
                processed_count = run_openface_processing(output_dir, move_files=True, specific_files=session_mirrored_files)
                
                if processed_count > 0:
                    messagebox.showinfo(
                        "OpenFace Processing Complete",
                        f"{processed_count} files from this session were processed with OpenFace and moved to the 'S1O Processed Files' directory."
                    )
                else:
                    messagebox.showinfo(
                        "OpenFace Processing",
                        "No files were processed with OpenFace."
                    )
            except Exception as e:
                messagebox.showerror(
                    "OpenFace Processing Error",
                    f"Error during OpenFace processing: {str(e)}"
                )
        # If openface_choice is "none", do nothing
        
        # Clean up
        root.destroy()

    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
