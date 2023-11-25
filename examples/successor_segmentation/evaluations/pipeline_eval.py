import os
import json
import sys
import subprocess
import argparse
from contextlib import contextmanager

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
        subprocess.run(["pip", "install", "-r", req_file], check=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()

def generate_config(base_directory):
    return {
        "evaluations": base_directory,
        "labels": f"{base_directory}/labels",
        "image_audio_pairs": f"{base_directory}/image_audio_pairs",
        "paired_evaluations": f"{base_directory}/paired_evaluations",
        "image_evaluations": f"{base_directory}/image_evaluations",
    }

def create_directories(config):
    for key, path in config.items():
        if not path.endswith(('.parquet', '.yaml')):
            os.makedirs(path, exist_ok=True)

def main():
    try:
        args = parse_args()
        config = {"local": generate_config("./evaluations")}
        selected_config = config[args.mode]
        create_directories(selected_config)
        # Update the path to the requirements file
        path = "./clip-video-encode/examples/successor_segmentation/evaluations/"
        result = install_requirements(path)
        if result != 0:
            return result
    except Exception as e:
        print(f"An exception occurred: {e}")
        return 1
    return 0  # Add this line

if __name__ == "__main__":
    sys.exit(main())  