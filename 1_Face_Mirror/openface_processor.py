#!/usr/bin/env python3
import os
import subprocess
import argparse

def process_videos(directory_path):
    """
    Processes all video files in the given directory that end with 'mirrored',
    ignoring files that end with 'debug'.
    
    Args:
        directory_path (str): Path to the directory containing video files
    """
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    
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

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process video files with OpenFace')
    parser.add_argument('--directory', type=str, 
                        help='Directory containing video files (default: "output" folder in script directory)',
                        default=None)
    
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
    process_videos(args.directory)

if __name__ == "__main__":
    main()
