#!/usr/bin/env python3
import os
import subprocess
import argparse
import shutil

def process_videos(directory_path):
    """
    Processes all video files in the given directory that end with 'mirrored',
    ignoring files that end with 'debug'.
    
    Args:
        directory_path (str): Path to the directory containing video files
    
    Returns:
        int: Number of files that were processed
    """
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return 0
    
    # OpenFace command path
    openface_cmd = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
    
    # Counter for processed files
    processed_count = 0
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(file_path):
            continue
        
        # Skip files with 'debug' in the filename
        if 'debug' in filename:
            print(f"Skipping debug file: {filename}")
            continue
        
        # Process files with 'mirrored' in the filename
        if 'mirrored' in filename:
            print(f"Processing file: {filename}")
            
            # Construct the command
            command = [
                openface_cmd,
                "-aus",
                "-verbose",
                "-tracked",
                "-f",
                file_path
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
    # Get the parent directory of the output directory
    parent_dir = os.path.dirname(os.path.dirname(output_dir))
    
    # Define the destination directory
    dest_dir = os.path.join(parent_dir, "1.5_Processed_Files")
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")
    
    # Counter for moved files
    moved_mov_count = 0
    moved_csv_count = 0
    
    # Move _rotated video files from output directory (handle both .mov and .MOV)
    for filename in os.listdir(output_dir):
        # Case-insensitive check for "_rotated.mov" ending
        if "_rotated.mov" in filename.lower():
            source_path = os.path.join(output_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                shutil.move(source_path, dest_path)
                moved_mov_count += 1
                print(f"Moved: {filename}")
            except Exception as e:
                print(f"Error moving {filename}: {e}")
    
    # Get the processed directory path (where OpenFace puts .csv files)
    # The processed directory is in the same folder as the script, not inside the output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, "processed")
    
    print(f"Looking for CSV files in: {processed_dir}")
    
    # Move .csv files if the processed directory exists
    if os.path.isdir(processed_dir):
        for filename in os.listdir(processed_dir):
            if filename.endswith(".csv"):
                source_path = os.path.join(processed_dir, filename)
                dest_path = os.path.join(dest_dir, filename)
                
                try:
                    shutil.move(source_path, dest_path)
                    moved_csv_count += 1
                    print(f"Moved: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
    else:
        print(f"Warning: Processed directory not found at {processed_dir}")
    
    print(f"\nFile moving complete. Moved {moved_mov_count} video files and {moved_csv_count} .csv files to {dest_dir}")
    return moved_mov_count, moved_csv_count

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process video files with OpenFace')
    parser.add_argument('--directory', type=str, 
                        help='Directory containing video files (default: "output" folder in script directory)',
                        default=None)
    parser.add_argument('--move-files', action='store_true',
                        help='Move processed files to 1.5_Processed_Files after completion')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no directory is specified, use the "output" folder in the same directory as the script
    if args.directory is None:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Set the default directory to "output" in the script directory
        default_dir = os.path.join(script_dir, "output")
        args.directory = default_dir
        print(f"Using default directory: {default_dir}")
    
    # Process videos in the specified directory
    processed_count = process_videos(args.directory)
    
    # Move the processed files if requested
    if args.move_files:
        move_processed_files(args.directory)

if __name__ == "__main__":
    main()
