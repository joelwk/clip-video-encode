import os
import pandas as pd
import json
import glob
import configparser
import subprocess
import argparse
import shutil
import sys
import ffmpeg
from srcs.pipeline import read_config, generate_config, install_local_package, parse_args
import cv2

def prepare_dataset_requirements(directories, external_parquet_path):
    if external_parquet_path is not None:
        # If an external Parquet file is provided, copy it to the directory
        shutil.copy(external_parquet_path, f"{directories}/dataset_requirements.parquet")
        print(f"Copied external Parquet file to {directories}")
    else:
        # Otherwise, create a new Parquet file from the default JSON data
        dataset_requirements = {
            "data": [
                {"url": "www.youtube.com/watch?v=nXBoOam5xJs", "caption": "The Deadly Portuguese Man O' War"},
                {"url": "www.youtube.com/watch?v=pYbbyuqv86Q", "caption": "Hate Speech is a marketing campaign for censorship"},
            ]
        }
        df = pd.DataFrame(dataset_requirements['data'])
        print(f"DataFrame to be saved:\n{df}")
        try:
            parquet_file_path = f"{directories}/dataset_requirements.parquet"
            df.to_parquet(parquet_file_path, index=False)
            print(f"Saved Parquet file at {parquet_file_path}")
        except Exception as e:
            print(f"Error while saving Parquet file: {e}")

def load_dataset_requirements(directory):
    # Read from the Parquet file instead of the JSON file
    return pd.read_parquet(f"{directory}/dataset_requirements.parquet").to_dict(orient='records')

def get_video_duration(video_file):
    vid_cap = cv2.VideoCapture(video_file)
    total_duration = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(vid_cap.get(cv2.CAP_PROP_FPS))
    vid_cap.release()
    return total_duration

def collect_video_metadata(video_files, output):
    keyframe_video_locs = []
    original_video_locs = []
    for video_file in video_files:
        video_id = os.path.basename(video_file).split('.')[0]
        json_meta_path = video_file.replace('.mp4', '.json')
        if not os.path.exists(json_meta_path):
            print(f"JSON metadata file does not exist: {json_meta_path}")
            continue
        with open(json_meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata for {video_id}: {metadata}")
            
        # Fallback to get_video_duration if 'duration' is not available in JSON metadata
        duration = metadata.get('video_metadata', {}).get('streams', [{}])[0].get('duration', None)
        if duration is None:
            print(f"Duration not found in metadata. Calculating duration for {video_id}.")
            duration = get_video_duration(video_file)
            
        keyframe_video_locs.append({
            "videoLoc": f"{output}/{video_id}_key_frames.mp4",
            "videoID": video_id,
            "duration": duration,
        })
        original_video_locs.append({
            "videoLoc": video_file,
            "videoID": video_id,
            "duration": duration,
        })
    return keyframe_video_locs, original_video_locs

def fix_codecs_in_directory(directory):
    video_files = glob.glob(f"{directory}/**/*.mp4", recursive=True)
    for video_file in video_files:
        video_id = os.path.basename(video_file).split('.')[0]
        input_file_path = video_file 
        output_file_path = os.path.join(directory, f"fixed_{video_id}.mp4")
        try:
            ffmpeg.input(input_file_path).output(output_file_path, vcodec='libx264', strict='-2', loglevel="quiet").overwrite_output().run(capture_stdout=True, capture_stderr=True)
            print(f"Successfully re-encoded {video_file}")
            os.remove(input_file_path)
            os.rename(output_file_path, input_file_path)
        except AttributeError as e:
            print(f"AttributeError: {e}. FFMPEG might not be correctly installed or imported.")
        except Exception as e:  
            print(f"An unexpected error occurred: {e}")

def segment_key_frames_in_directory(directory, output_directory):
    video_files = glob.glob(f"{directory}/**/*.mp4", recursive=True)
    for video_file in video_files:
        video_id = os.path.basename(video_file).split('.')[0]
        input_file = video_file
        output_file = os.path.join(output_directory, f"{video_id}_key_frames.mp4")
        print(f"Segmenting key frames for {video_id}...")
        command = f'ffmpeg -y -loglevel error -discard nokey -i {input_file} -c:s copy -c copy -copyts {output_file}'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print(f"Successfully segmented key frames for {video_id}.")
        else:
            print(f"Failed to segment key frames for {video_id}. Error: {stderr.decode('utf8')}")

def save_metadata_to_parquet(keyframe_video_locs, original_video_locs, directory):
    keyframe_video_df = pd.DataFrame(keyframe_video_locs)
    original_video_df = pd.DataFrame(original_video_locs)
    keyframe_video_df['duration'] = keyframe_video_df['duration'].astype(float)
    original_video_df['duration'] = original_video_df['duration'].astype(float)
    keyframe_video_df.to_parquet(f'{directory}/keyframe_video_requirements.parquet', index=False)
    original_video_df.to_parquet(f'{directory}/original_video_requirements.parquet', index=False)

def prepare_clip_encode(directory, output, original_videos):
    dataset_requirements = load_dataset_requirements(directory)
    df = pd.DataFrame(dataset_requirements)
    video_files = glob.glob(f"{original_videos}/**/*.mp4", recursive=True)
    keyframe_video_locs, original_video_locs = collect_video_metadata(video_files, output)
    save_metadata_to_parquet(keyframe_video_locs, original_video_locs, directory)

def run_video2dataset_with_yt_dlp(directory, output):
    os.makedirs(output, exist_ok=True)
    url_list = f'{directory}/dataset_requirements.parquet'
    print(f"Reading URLs from: {url_list}")
    df = pd.read_parquet(url_list)
    for idx, row in df.iterrows():
        print(f"Processing video {idx+1}: {row['url']}")
        command = [
            'video2dataset',
            '--input_format', 'parquet',
            '--url_list', url_list,
            '--encode_formats', '{"video": "mp4", "audio": "m4a"}',
            '--output_folder', output,
            '--config', './clip-video-encode/examples/successor_segmentation/config.yaml'] 
        result = subprocess.run(command, capture_output=True, text=True)
        print("Return code:", result.returncode)
        print("STDOUT:", result.stdout)
        
def main():
    directories = read_config(section="directory")
    external_parquet = directories.get("external_parquet", None)
    if external_parquet == "None":
        external_parquet = None
    prepare_dataset_requirements(directories["base_directory"], external_parquet)
    run_video2dataset_with_yt_dlp(directories["base_directory"], directories["originalframes"])
    fix_codecs_in_directory(directories["originalframes"])
    segment_key_frames_in_directory(directories["originalframes"], directories["keyframes"])
    prepare_clip_encode(directories["base_directory"], directories["keyframes"], directories["originalframes"])
    install_local_package('./clip-video-encode')
    exit_status = 0 
    print(f"Exiting {__name__} with status {exit_status}")
    return exit_status

if __name__ == "__main__":
    exit_status = main()
    sys.exit(exit_status)