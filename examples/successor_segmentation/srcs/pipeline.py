import os
import json
import sys
import subprocess
import argparse
from contextlib import contextmanager
import configparser

def read_config(section="directory", config_path='./clip-video-encode/examples/successor_segmentation/config.ini'):
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(config_path)
    config = configparser.ConfigParser()
    config.read(config_path)
    if section not in config.sections():
        print(f"Section {section} not found in configuration.")
        raise KeyError(section)
    return {key: config[section][key] for key in config[section]}
    
def string_to_bool(string_value):
    return string_value.lower() in ['true', '1', 't', 'y', 'yes', 'on']

@contextmanager
def change_directory(destination):
    original_path = os.getcwd()
    if not os.path.exists(destination):
        os.makedirs(destination)
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(original_path)

def install_requirements(directory):
    req_file = os.path.join(directory, 'requirements.txt')
    if os.path.exists(req_file):
        result = subprocess.run(["pip", "install", "-r", req_file], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error installing requirements: {result.stderr}")
            return 1

def install_local_package(directory):
    with change_directory(directory):
        result = subprocess.run(["pip", "install", "-e", "."], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error installing local package: {result.stderr}")
            return 1

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()
    
def generate_config(base_directory):
    return {
        "directory": base_directory,
        "original_videos": f"{base_directory}/originalvideos",
        "keyframe_videos": f"{base_directory}/keyframes",
        "embedding_output": f"{base_directory}/originalembeddings",
        "keyframe_embedding_output": f"{base_directory}/keyframeembeddings",
        "config_yaml": f"{base_directory}/config.yaml"
    }

def delete_associated_files(video_id: int, config):
    try:
        file_paths = [
            # remove original files from video2dataset
            f"./{config['originalframes']}/{video_id}.m4a",
            f"./{config['originalframes']}/{video_id}.mp4",
            f"./{config['originalframes']}/{video_id}.json",
            # remove json and numpy files from clip-video-encode
             f"./{config['embeddings']}/{video_id}.json",
             f"./{config['embeddings']}/{video_id}.npy",
            
            f"./{config['keyframes']}/{video_id}.mp4",
            f"./{config['keyframe_outputs']}/{video_id}",
            f"./{config['keyframe_audio_clip_output']}/{video_id}",
        ]
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error in deleting files for video ID {video_id}: {e}")

def create_directories(config):
    for key, path in config.items():
        if not path.endswith(('.parquet', '.yaml')):
            os.makedirs(path, exist_ok=True)

def clone_repository(git_url, target_dir):
    repo_name = git_url.split("/")[-1].replace(".git", "")
    full_path = os.path.join(target_dir, repo_name)
    if not os.path.exists(full_path):
        result = subprocess.run(["git", "clone", git_url, full_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error cloning repository: {result.stderr}")
            return 1
    return full_path
    
def modify_requirements_txt(file_path, target_packages):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(file_path, 'w') as f:
        for line in lines:
            modified = False
            for package, new_version in target_packages.items():
                if line.startswith(package):
                    f.write(f"{package}{new_version}\n")
                    modified = True
                    break
            if not modified:
                f.write(line)

def main():
    directories =  read_config(section="directory")
    try:
        args = parse_args()
        config = {"local": generate_config(directories['base_directory'])}
        selected_config = config[args.mode]
        create_directories(selected_config)
        video2dataset_path = clone_repository("https://github.com/iejMac/video2dataset.git", "./repos")
        print('installing target packages')
        target_packages = {
            "pandas": ">=1.1.5,<2",
            "pyarrow": ">=6.0.1,<8",
            "imageio-ffmpeg": ">=0.4.0,<1",
            "transformers": ">=4.35.0",
        }
        modify_requirements_txt(f"{video2dataset_path}/requirements.txt", target_packages)
        with open(f"{video2dataset_path}/requirements.txt", "a") as f:
            f.write("imagehash>=4.3.1\n")
        status = install_local_package(video2dataset_path)
        if status and status != 0:
            return status
        return 0 
    except Exception as e:
        print(f"An exception occurred: {e}")
        return 1  

if __name__ == "__main__":
    sys.exit(main())