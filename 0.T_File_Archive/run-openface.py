#!/usr/bin/env python3
"""
Main script to run OpenFace processing on video files.
This script will automatically process videos in the "output" folder
that is in the same directory as this script.
"""

from openface_processor import process_videos, move_processed_files
import os
import argparse

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process video files with OpenFace and move processed files')
    parser.add_argument('--no-move', action='store_true',
                        help='Skip moving processed files after completion')
    
    # Parse arguments
    args = parser.parse_args()
    
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
    processed_count = process_videos(target_dir)
    
    # Move processed files unless specifically told not to
    # Note: We'll still try to move files even if no new processing happened
    # This is useful if you just want to move files from a previous run
    if not args.no_move:
        print("\nMoving processed files...")
        move_processed_files(target_dir)

if __name__ == "__main__":
    main()
