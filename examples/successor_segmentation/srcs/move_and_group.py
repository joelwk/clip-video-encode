import os
import glob
import shutil
from srcs.load_data import read_config

def move_and_group_files(directories):
    # Define source directories for various categories of files
    src_dirs = {
        'originalvideos': directories['originalframes'],
        'keyframevideos': directories['keyframes'],
        'keyframeembeddings': directories['embeddings'],
        'keyframe_clips': directories['keyframe_clips_output'],
        'keyframe_audio_clips': directories['keyframe_audio_clip_output'],
        'keyframes': directories['keyframe_outputs'],
        'keyframe_clip_embeddings': directories['keyframe_clip_embeddings_outputs'],
    }

    # Define the destination directory where files will be moved to
    dest_dir = './completedatasets'
    os.makedirs(dest_dir, exist_ok=True)
    # Initialize a dictionary to store integer suffixes for each category
    integer_suffixes = {}
    # Loop through source directories and list files
    for category, src_directory in src_dirs.items():
        for file_path in glob.glob(f"{src_directory}/*"):
            basename = os.path.basename(file_path)
            # Skip any files that are statistics files
            if basename.endswith("_stats"):
                continue
            # Extract integer suffix from file name
            integer_suffix = basename.split('.')[0]
            # For some categories, there are nested files; process them
            if category in ['keyframes', 'keyframe_clips', 'keyframe_clip_embeddings','keyframe_audio_clips']:
                for nested_file in glob.glob(f"{file_path}/*"):
                    if integer_suffix not in integer_suffixes:
                        integer_suffixes[integer_suffix] = []
                    integer_suffixes[integer_suffix].append((category, nested_file))
            else:
                if integer_suffix not in integer_suffixes:
                    integer_suffixes[integer_suffix] = []
                integer_suffixes[integer_suffix].append((category, file_path))
                
    # Create destination directories and move files
    for integer_suffix, file_tuples in integer_suffixes.items():
        integer_dest_dir = os.path.join(dest_dir, integer_suffix)
        os.makedirs(integer_dest_dir, exist_ok=True)
        for category, file_path in file_tuples:
            category_dest_dir = os.path.join(integer_dest_dir, category)
            os.makedirs(category_dest_dir, exist_ok=True)
            # Move the file to the new location
            new_file_path = os.path.join(category_dest_dir, os.path.basename(file_path))
            shutil.move(file_path, new_file_path)
            print(f"Moved {file_path} to {new_file_path}")

def cleanup_unwanted_dirs(directory, unwanted_dirs=None):
    if unwanted_dirs:
        for unwanted_dir in unwanted_dirs:
            path_to_remove = os.path.join(directory, unwanted_dir)
            if os.path.exists(path_to_remove):
                shutil.rmtree(path_to_remove)
                print(f"Removed {path_to_remove}")
    else:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed entire directory: {directory}")

def main():
    directories = read_config()
    move_and_group_files(directories)
    cleanup_unwanted_dirs('./completedatasets', ['00000_stats', '00000'])
    cleanup_unwanted_dirs(directories['output'])  
    cleanup_unwanted_dirs(directories['base_directory'])  

if __name__ == "__main__":
    main()