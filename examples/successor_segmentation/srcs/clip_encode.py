from srcs.pipeline import read_config
import subprocess
import argparse

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

    with change_dir('./clip-video-encode/'):
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
    clipencode_path = './clip-video-encode/'
    install_requirements(clipencode_path)
    clip_encode(directories)
    return 0

if __name__ == "__main__":
    main()