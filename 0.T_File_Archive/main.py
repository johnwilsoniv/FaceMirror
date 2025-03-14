import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from pathlib import Path
from stable_face_core import StableFaceSplitter

def main():
    """Command-line interface with enhanced stability analysis"""
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

        # Ask if the user wants to enable stability analysis
        analyze_stability = messagebox.askyesno(
            "Stability Analysis", 
            "Would you like to generate stability analysis graphs?\n\n"
            "This requires pandas and matplotlib to be installed."
        )

        splitter = StableFaceSplitter(debug_mode=True, log_midline=True)
        results = []

        for input_path in input_paths:
            try:
                outputs = splitter.process_video(input_path, output_dir, analyze_stability)
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
                log_file = next((f for f in result['outputs'] if 'midline_log.csv' in f), None)
                stability_image = next((f for f in result['outputs'] if 'midline_log.png' in f), None)

                if left_video:
                    summary += f"  - Left: {Path(left_video).name}\n"
                if right_video:
                    summary += f"  - Right: {Path(right_video).name}\n"
                if debug_video:
                    summary += f"  - Debug: {Path(debug_video).name}\n"
                if rotated_video:
                    summary += f"  - Rotated: {Path(rotated_video).name}\n"
                if log_file:
                    summary += f"  - Log: {Path(log_file).name}\n"
                if stability_image:
                    summary += f"  - Stability Analysis: {Path(stability_image).name}\n"
            else:
                summary += f"✗ {Path(result['input']).name} - Error: {result['error']}\n"

        messagebox.showinfo("Processing Complete", summary)

    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
