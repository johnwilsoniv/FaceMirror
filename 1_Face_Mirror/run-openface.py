#!/usr/bin/env python3
"""
Main script to run OpenFace processing on video files.
This script will automatically process videos in the "output" folder
that is in the same directory as this script.
"""

from openface_processor import process_videos
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the target directory to "output" in the script directory
    target_dir = os.path.join(script_dir, "output")
    
    print(f"Looking for video files in: {target_dir}")
    
    # Check if the output directory exists
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        print("Please create an 'output' folder in the same directory as this script.")
        return
    
    # Process videos in the output directory
    process_videos(target_dir)

if __name__ == "__main__":
    main()
