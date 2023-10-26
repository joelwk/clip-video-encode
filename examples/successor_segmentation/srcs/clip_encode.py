from srcs.pipeline import generate_config
import os
import subprocess
import argparse

def install_requirements(directory):
    req_file = os.path.join(directory, 'requirements.txt')
    if os.path.exists(req_file):
        subprocess.run(["pip", "install", "-r", req_file], check=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()

def clip_encode(selected_config):
    from contextlib import contextmanager

    @contextmanager
    def change_dir(target):
        od = os.getcwd()
        os.chdir(target)
        yield
        os.chdir(od)

    with change_dir('./clip-video-encode/'):
        from clip_video_encode import clip_video_encode

    clip_video_encode(
        selected_config["keyframe_parquet"],
        selected_config["keyframe_embedding_output"],
        frame_workers=25,
        take_every_nth=1,
        metadata_columns=['videoLoc', 'videoID', 'duration']
    )

def main():
    args = parse_args()
    config = {"local": generate_config("./datasets")}  
    selected_config = config[args.mode]
    clipencode_path = './clip-video-encode/'
    install_requirements(clipencode_path)
    clip_encode(selected_config)
    return 0

if __name__ == "__main__":
    main()
