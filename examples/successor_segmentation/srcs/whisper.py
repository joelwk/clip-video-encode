import os
import sys
import json
import subprocess
import glob
from srcs.segment_processing import read_thresholds_config

def install_requirements():
    try:
        import torch
        import pydub
        import transformers
    except ImportError:
        print("Installing required packages and restarting...")
        subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio"])
        subprocess.run(["pip", "install", "accelerate", "optimum"])
        subprocess.run(["pip", "install", "ipython-autotime"])
        subprocess.run(["pip", "install", "pydub"])
install_requirements()

import torch
from pydub import AudioSegment
from transformers import pipeline

def convert_audio_files(input_directory, output_directory, output_format="flac"):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(input_directory):
        if filename.endswith(".m4a"):
            m4a_path = os.path.join(input_directory, filename)
            output_filename = os.path.splitext(filename)[0] + f".{output_format}"
            output_path = os.path.join(output_directory, output_filename)
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Overwriting.")
            audio = AudioSegment.from_file(m4a_path, format="m4a")
            audio.export(output_path, format=output_format)
            print(f"Converted {m4a_path} to {output_path}")

def segment_audio_using_keyframes(audio_path, audio_clip_output_dir, keyframe_timestamps, thresholds, suffix_=None):
    start_time = 0
    segment_idx = 0
    for end_time in keyframe_timestamps:
        adjusted_start_time, adjusted_end_time = start_time, end_time
        if "keyframe" in str(suffix_):
            tolerance = float(thresholds['tolerance']) * (end_time - start_time)
            adjusted_start_time += tolerance
            adjusted_end_time -= tolerance
        output_path = f"{audio_clip_output_dir}/keyframe_audio_clip_{segment_idx}_{suffix_}.flac"
        command = [
            'ffmpeg',
            '-ss', str(adjusted_start_time),
            '-to', str(adjusted_end_time),
            '-i', audio_path,
            '-acodec', 'flac',
            '-y', output_path
        ]
        subprocess.run(command)
        start_time = end_time
        segment_idx += 1 
    # Save timestamps to JSON
    json_path = os.path.join(audio_clip_output_dir, 'timestamps.json')
    with open(json_path, 'w') as f:
        json.dump(keyframe_timestamps, f)

def audio_pipeline(audio_path, audio_clip_output_dir, thresholds, chunk_length):
    try:
        pipe = pipeline("automatic-speech-recognition",
                        "openai/whisper-large-v2",
                        torch_dtype=torch.float16,
                        device="cuda:0")
        outputs = pipe(audio_path,
                       chunk_length_s=chunk_length,
                       batch_size=24,
                       return_timestamps=True)
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return
    chunks = outputs.get("chunks", [])
    if not chunks:
        print("No chunks returned by the pipeline.")
        return
    output_aligned = []
    for idx, chunk in enumerate(chunks):
        segment_info = {
            'segment_idx': idx,
            'timestamp': chunk.get('timestamp', (None, None)),
            'text': chunk.get('text', '')
        }
        output_aligned.append(segment_info)
    json_path = os.path.join(audio_clip_output_dir, 'outputs.json')
    with open(json_path, 'w') as f:
        json.dump(output_aligned, f)
    try:
        timestamps = sorted(set([time_point for chunk in outputs["chunks"] for time_point in chunk['timestamp'] if time_point is not None]))
    except Exception as e:
        print(f"Error while sorting timestamps: {e}")
        return
    try:
        segment_audio_using_keyframes(audio_path, audio_clip_output_dir, timestamps, thresholds, suffix_="whisper")
    except Exception as e:
        print(f"Error in segment_audio_using_keyframes: {e}")
        
def process_audio_files():
    try:
        thresholds = read_thresholds_config()
        base_path = './completedatasets/'
        for n in os.listdir(base_path):
            initial_input_directory = os.path.join(base_path, n, 'originalvideos')
            audio_clip_output_dir = os.path.join(base_path, n, 'keyframe_audio_clips', 'whisper_audio_segments')
            parent_dir = f'/content/completedatasets/{n}/keyframe_audio_clips'
            total_chunks = len(glob.glob(f"{parent_dir}/*.m4a"))
            if not os.path.exists(audio_clip_output_dir):
                os.makedirs(audio_clip_output_dir)
            for audio_file in os.listdir(initial_input_directory):
                if audio_file.endswith('.m4a'):
                    individual_output_dir = os.path.join(audio_clip_output_dir, os.path.splitext(audio_file)[0])
                    if not os.path.exists(individual_output_dir):
                        os.makedirs(individual_output_dir)
                    convert_audio_files(initial_input_directory, individual_output_dir)
                    flac_file = os.path.splitext(audio_file)[0] + '.flac'
                    audio_path = os.path.join(individual_output_dir, flac_file)
                    audio_pipeline(audio_path, individual_output_dir, thresholds, total_chunks)
    except Exception as e:
        print(f"An exception occurred: {e}")