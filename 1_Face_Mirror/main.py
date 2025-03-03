import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from pathlib import Path
from face_splitter import StableFaceSplitter
from openface_integration import run_openface_processing

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
            return

        output_dir = Path.cwd() / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        splitter = StableFaceSplitter(debug_mode=True)
        results = []

        for input_path in input_paths:
            try:
                outputs = splitter.process_video(input_path, output_dir)
                results.append({
                    'input': input_path,
                    'success': True,
                    'outputs': outputs
                })
            except Exception as e:
                results.append({
                    'input': input_path,
                    'success': False,
                    'error': str(e)
                })

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
        
        # Ask if user wants to run OpenFace processing
        run_openface = messagebox.askyesno(
            "OpenFace Processing", 
            "Would you like to run OpenFace processing on the mirrored videos?\n\n"
            "This will process the mirrored videos with OpenFace to extract facial features."
        )
        
        if run_openface:
            try:
                # Always move files after processing (move_files=True)
                processed_count = run_openface_processing(output_dir, move_files=True)
                
                if processed_count > 0:
                    messagebox.showinfo(
                        "OpenFace Processing Complete",
                        f"{processed_count} files were processed with OpenFace and moved to the '1.5_Processed_Files' directory."
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

    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
