import os
import json
import sys
import subprocess
import argparse
from contextlib import contextmanager

def install_requirements(req_file):
    if os.path.exists(req_file):
        result = subprocess.run(["pip", "install", "-r", req_file], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error installing requirements: {result.stderr}")
            return 1


def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()

def generate_config(base_directory):
    return {
        "evaluations": base_directory,
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
        # Assuming requirements.txt is in the root directory of your project
        install_requirements("requirements.txt")
    except Exception as e:
        print(f"An exception occurred: {e}")
        return 1  
if __name__ == "__main__":
    sys.exit(main())  

