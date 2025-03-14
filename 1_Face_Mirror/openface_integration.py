#!/usr/bin/env python3
import os
import subprocess
import shutil
from pathlib import Path

def process_videos(directory_path, specific_files=None):
    """
    Processes video files in the given directory that end with 'mirrored',
    ignoring files that end with 'debug'.
    
    Args:
        directory_path (str): Path to the directory containing video files
        specific_files (list, optional): List of specific files to process. If None, all eligible files in the directory will be processed.
    
    Returns:
        int: Number of files that were processed
    """
    directory_path = Path(directory_path)
    
    # Check if directory exists
    if not directory_path.is_dir():
        print(f"Error: Directory '{directory_path}' does not exist.")
        return 0
    
    # OpenFace command path
    openface_cmd = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
    
    # Counter for processed files
    processed_count = 0
    
    # Define which files to process
    files_to_process = []
    
    if specific_files:
        # Process only the specific files
        files_to_process = [Path(f) for f in specific_files]
        print(f"Processing {len(files_to_process)} specific files from current session.")
    else:
        # Process all eligible files in the directory
        files_to_process = list(directory_path.iterdir())
        print(f"Processing all eligible files in {directory_path}")
    
    # Process each file
    for file_path in files_to_process:
        # Skip if not a file or doesn't exist
        if not file_path.is_file():
            print(f"Warning: {file_path} does not exist or is not a file. Skipping.")
            continue
        
        filename = file_path.name
        
        # Skip files with 'debug' in the filename
        if 'debug' in filename:
            print(f"Skipping debug file: {filename}")
            continue
        
        # Process file with 'mirrored' in the filename
        if 'mirrored' in filename:
            print(f"Processing file: {filename}")
            
            # Construct the command
            command = [
                openface_cmd,
                "-aus",
                "-verbose",
                "-tracked",
                "-f",
                str(file_path)
            ]
            
            try:
                # Run the OpenFace command
                subprocess.run(command, check=True)
                processed_count += 1
                print(f"Successfully processed: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"\nProcessing complete. {processed_count} files were processed.")
    return processed_count

def move_processed_files(output_dir):
    """
    Moves _rotated.mov files from the output directory and .csv files from the processed 
    directory to a '1.5_Processed_Files' folder one level up from the parent directory.
    Now handles both lowercase .mov and uppercase .MOV extensions.
    
    Args:
        output_dir (str): Path to the output directory containing the video files
    
    Returns:
        tuple: (int, int) Number of moved video files and .csv files
    """
    output_dir = Path(output_dir)
    
    # Get the parent directory of the output directory
    parent_dir = output_dir.parent.parent
    
    # Define the destination directory
    dest_dir = parent_dir / "1.5_Processed_Files"
    
    # Create the destination directory if it doesn't exist
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
        print(f"Created destination directory: {dest_dir}")
    
    # Counter for moved files
    moved_mov_count = 0
    moved_csv_count = 0
    
    # Move _rotated video files from output directory (handle both .mov and .MOV)
    for file_path in output_dir.iterdir():
        filename = file_path.name
        # Case-insensitive check for "_rotated.mov" ending
        if "_rotated.mov" in filename.lower():
            dest_path = dest_dir / filename
            
            try:
                shutil.move(str(file_path), str(dest_path))
                moved_mov_count += 1
                print(f"Moved: {filename}")
            except Exception as e:
                print(f"Error moving {filename}: {e}")
    
    # Get the processed directory path (where OpenFace puts .csv files)
    # The processed directory is in the same folder as the script, not inside the output directory
    script_dir = Path(__file__).parent
    processed_dir = script_dir / "processed"
    
    print(f"Looking for CSV files in: {processed_dir}")
    
    # Move .csv files if the processed directory exists
    if processed_dir.is_dir():
        for file_path in processed_dir.iterdir():
            if file_path.suffix.lower() == '.csv':
                filename = file_path.name
                dest_path = dest_dir / filename
                
                try:
                    shutil.move(str(file_path), str(dest_path))
                    moved_csv_count += 1
                    print(f"Moved: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
    else:
        print(f"Warning: Processed directory not found at {processed_dir}")
    
    print(f"\nFile moving complete. Moved {moved_mov_count} video files and {moved_csv_count} .csv files to {dest_dir}")
    return moved_mov_count, moved_csv_count

def run_openface_processing(output_dir, move_files=True, specific_files=None):
    """
    Main function to run OpenFace processing on video files in the output directory.
    
    Args:
        output_dir (str): Path to the output directory containing video files
        move_files (bool): Whether to move processed files after completion
        specific_files (list, optional): List of specific files to process. If None, all eligible files in the output_dir will be processed.
    
    Returns:
        int: Number of processed files
    """
    output_dir = Path(output_dir)
    
    print(f"Looking for video files in: {output_dir}")
    
    # Check if the output directory exists
    if not output_dir.is_dir():
        print(f"Error: Directory '{output_dir}' does not exist.")
        return 0
    
    # Process videos in the output directory
    processed_count = process_videos(output_dir, specific_files)
    
    # Move processed files if requested
    if move_files:
        print("\nMoving processed files...")
        move_processed_files(output_dir)
    
    return processed_count
