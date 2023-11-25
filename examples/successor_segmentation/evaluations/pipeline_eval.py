import os
import json
import sys
import subprocess
import argparse
from contextlib import contextmanager


def install_requirements():
    try:
        import open_clip
    except ImportError:
        print("Installing required packages and restarting...")
        subprocess.run(["pip", "install", "yt-dlp"])
        subprocess.run(["pip", "install", "scikit-learn==1.3.0"])
        subprocess.run(["pip", "install", "pydub"])


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
        install_requirements()
    except Exception as e:
        print(f"An exception occurred: {e}")
        return 1
    return 0 

if __name__ == "__main__":
    sys.exit(main()) 