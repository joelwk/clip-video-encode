import os
import argparse
import subprocess
from pipeline import generate_config 

def install_requirements(directory):
    req_file = os.path.join(directory, 'requirements.txt')
    if os.path.exists(req_file):
        subprocess.run(["pip", "install", "-r", req_file], check=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()

def clip_encode(selected_config):
    od = os.getcwd()
    os.chdir('./clip-video-encode/')
    from clip_video_encode import clip_video_encode
    os.chdir(od)
    dataset_parquet_path = "./datasets/keyframe_video_requirements.parquet"

    clip_video_encode(
        selected_config["keyframe_parquet"],
        selected_config["keyframe_embedding_output"],
        frame_workers=25,
        take_every_nth=1,
        metadata_columns=['videoLoc', 'videoID', 'duration']
    )

if __name__ == "__main__":
    args = parse_args()
    config = {"local": generate_config("./datasets")}  
    selected_config = config[args.mode]

    clipencode_path = './clip-video-encode/'

    install_requirements(clipencode_path)

    clip_encode(selected_config)
    
