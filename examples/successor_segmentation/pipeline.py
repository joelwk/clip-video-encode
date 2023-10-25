import os
import subprocess
import argparse
import sys
from contextlib import contextmanager

def install_requirements(directory):
    req_file = os.path.join(directory, 'requirements.txt')
    if os.path.exists(req_file):
        subprocess.run(["pip", "install", "-r", req_file])

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

def install_local_package(directory):
    original_directory = os.getcwd()
    if os.path.exists(directory):
        os.chdir(directory)
        subprocess.run(["pip", "install", "-e", "."])
        os.chdir(original_directory)
    else:
        print(f"Directory {directory} does not exist. Skipping package installation.")
        
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
        "keyframe_parquet": f"{base_directory}/keyframe_video_requirements.parquet",
        "config_yaml": f"{base_directory}/config.yaml"
    }

def create_directories(config):
    for key, path in config.items():
        if not path.endswith(('.parquet', '.yaml')):
            os.makedirs(path, exist_ok=True)

def clone_repository(git_url, target_dir):
    repo_name = git_url.split("/")[-1].replace(".git", "")
    full_path = os.path.join(target_dir, repo_name)
    if not os.path.exists(full_path):
        subprocess.run(["git", "clone", git_url, full_path])
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

if __name__ == "__main__":
    args = parse_args()
    config = {"local": generate_config("./datasets")}
    selected_config = config[args.mode]
    create_directories(selected_config)

    video2dataset_path = clone_repository("https://github.com/iejMac/video2dataset.git", "./repos")
    target_packages = {
        "pandas": ">=1.1.5,<2",
        "pyarrow": ">=6.0.1,<8",
        "imageio-ffmpeg": ">=0.4.0,<1",
    }

    modify_requirements_txt(f"{video2dataset_path}/requirements.txt", target_packages)
    with open(f"{video2dataset_path}/requirements.txt", "a") as f:
        f.write("imagehash>=4.3.1\n")
    install_local_package(video2dataset_path)

