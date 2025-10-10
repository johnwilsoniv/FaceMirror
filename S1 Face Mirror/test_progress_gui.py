#!/usr/bin/env python3
"""
Test script for the progress window GUI.
This simulates video processing progress without actually processing videos.
"""

import tkinter as tk
import time
from progress_window import ProcessingProgressWindow, ProgressUpdate


def test_progress_window():
    """Test the progress window with simulated video processing"""
    print("Starting progress window test...")

    # Create root window (hidden)
    root = tk.Tk()
    root.withdraw()

    # Test with 3 videos
    total_videos = 3
    window = ProcessingProgressWindow(total_videos=total_videos)

    # Simulate processing each video
    for video_num in range(1, total_videos + 1):
        video_name = f"patient_{video_num:03d}_facial_exam.mp4"
        print(f"\nSimulating video {video_num}/{total_videos}: {video_name}")

        # Stage 1: Reading frames
        print("  - Reading frames...")
        total_frames = 500
        for i in range(0, total_frames + 1, 10):
            window.update_progress(ProgressUpdate(
                video_name=video_name,
                video_num=video_num,
                total_videos=total_videos,
                stage='reading',
                current=i,
                total=total_frames,
                message="Reading frames into memory..."
            ))
            time.sleep(0.02)  # Simulate work

        # Stage 2: Processing frames
        print("  - Processing frames...")
        for i in range(0, total_frames + 1, 5):
            window.update_progress(ProgressUpdate(
                video_name=video_name,
                video_num=video_num,
                total_videos=total_videos,
                stage='processing',
                current=i,
                total=total_frames,
                message="Processing frames with face detection..."
            ))
            time.sleep(0.03)  # Simulate work (processing is slower)

        # Stage 3: Writing output
        print("  - Writing output...")
        for i in range(0, total_frames + 1, 10):
            window.update_progress(ProgressUpdate(
                video_name=video_name,
                video_num=video_num,
                total_videos=total_videos,
                stage='writing',
                current=i,
                total=total_frames,
                message="Writing frames to output files..."
            ))
            time.sleep(0.02)  # Simulate work

        # Complete
        window.update_progress(ProgressUpdate(
            video_name=video_name,
            video_num=video_num,
            total_videos=total_videos,
            stage='complete',
            current=total_frames,
            total=total_frames,
            message="Video processing complete"
        ))
        print("  - Complete!")
        time.sleep(0.5)  # Brief pause between videos

    print("\nAll videos processed. Closing window in 2 seconds...")
    time.sleep(2)

    # Close progress window
    window.close()
    root.destroy()

    print("Test complete!")


if __name__ == "__main__":
    test_progress_window()
