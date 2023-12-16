from srcs.pipeline import read_config
import subprocess
import argparse
import os

def install_requirements(directory):
    req_file = os.path.join(directory, 'requirements.txt')
    if os.path.exists(req_file):
        subprocess.run(["pip", "install", "-r", req_file], check=True)

def clip_encode(selected_config):
    from contextlib import contextmanager

    @contextmanager
    def change_dir(target):
        od = os.getcwd()
        os.chdir(target)
        yield
        os.chdir(od)

    # Dynamically set the base path
    if 'content' in os.getcwd():
        base_path = '/content'  # Google Colab
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # Local environment
    # Construct the absolute path for clip-video-encode
    clipencode_abs_path = os.path.join(base_path, 'clip-video-encode')
    with change_dir(clipencode_abs_path):
        from clip_video_encode import clip_video_encode
    clip_video_encode(
        f'{selected_config["base_directory"]}/keyframe_video_requirements.parquet',
        selected_config["embeddings"],
        frame_workers=25,
        take_every_nth=1,
        metadata_columns=['videoLoc', 'videoID', 'duration']
    )

def main():
    directories = read_config(section="directory")
    # Same dynamic base path logic
    if 'content' in os.getcwd():
        base_path = '/content'
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    clipencode_path = os.path.join(base_path, 'clip-video-encode')
    install_requirements(clipencode_path)
    clip_encode(directories)
    return 0

if __name__ == "__main__":
    main()
