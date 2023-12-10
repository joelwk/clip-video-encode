import os
import sys
import json
import subprocess
import glob
import warnings
from srcs.load_data import read_config, string_to_bool

warnings.filterwarnings("ignore", category=UserWarning)

def read_keyframe_data(keyframe_json_path):
    with open(keyframe_json_path, 'r') as file:
        return json.load(file)
        
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
        subprocess.run(["pip", "install", "--upgrade","transformers"])

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

def segment_audio_using_keyframes(audio_path, audio_clip_output_dir, keyframe_data, duration, suffix_=None):
    os.makedirs(audio_clip_output_dir, exist_ok=True)
    output_aligned = [{'segment_idx': idx, 'timestamp': [keyframe['time_frame'], keyframe['time_frame'] + duration]} for idx, keyframe in keyframe_data.items()]
    for segment in output_aligned:
        start_time = segment['timestamp'][0]
        adjusted_start_time = start_time
        suffix_str = f"_{suffix_}" if suffix_ else ""
        output_segment_path = f"{audio_clip_output_dir}/segment_{segment['segment_idx']}{suffix_str}.flac"
        command = [
            'ffmpeg',
            '-ss', str(adjusted_start_time),
            '-t', str(duration),
            '-i', audio_path,
            '-acodec', 'flac',
            '-y', output_segment_path
        ]
        subprocess.run(command, check=True)
    json_path = os.path.join(audio_clip_output_dir, 'keyframe_timestamps.json')
    with open(json_path, 'w') as f:
        json.dump(output_aligned, f)

def audio_pipeline(audio_path, audio_clip_output_dir, keyframe_data, duration):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_path)
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe = pipeline("automatic-speech-recognition",
                        "openai/whisper-large-v2",
                        torch_dtype=torch.float16,
                        device=device)

        output_aligned_final = []

        # Process each segment defined in keyframe_data
        for idx, keyframe in keyframe_data.items():
            start_time_ms = keyframe['time_frame'] * 1000  # Convert start time to milliseconds
            end_time_ms = start_time_ms + (duration * 1000)  # Calculate end time in milliseconds

            # Extract the segment
            audio_segment = audio[start_time_ms:end_time_ms]

            # Save this segment to a temporary file
            temp_path = os.path.join(audio_clip_output_dir, f'temp_segment_{idx}.flac')
            audio_segment.export(temp_path, format='flac')

            # Process the segment using Whisper
            outputs = pipe(temp_path, return_timestamps=True)
            os.remove(temp_path)  # Remove the temporary file
            chunks = outputs.get("chunks", [])
            if chunks:
                transcript = ' '.join(chunk.get('text', '') for chunk in chunks)
                segment_info = {
                    'segment_idx': idx,
                    'timestamp': [start_time_ms / 1000, end_time_ms / 1000],  
                    'text': transcript}
                output_aligned_final.append(segment_info)
                
        # Save the results to a JSON file
        json_path = os.path.join(audio_clip_output_dir, 'outputs.json')
        with open(json_path, 'w') as f:
            json.dump(output_aligned_final, f)
    except Exception as e:
        print(f"Error in audio_pipeline: {e}")

def full_audio_transcription_pipeline(audio_path, output_dir):
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe = pipeline("automatic-speech-recognition",
                        "openai/whisper-large-v2",
                        torch_dtype=torch.float16,
                        device=device)

        # Process the entire audio file using Whisper
        outputs = pipe(audio_path,chunk_length_s=30,batch_size=30, return_timestamps=True)
        chunks = outputs.get("chunks", [])
        if not chunks:
            print(f"No chunks returned by the pipeline for {audio_path}.")
            return
        transcript = ' '.join(chunk.get('text', '') for chunk in chunks)
        full_transcript_path = os.path.join(output_dir, 'full_transcript.json')
        with open(full_transcript_path, 'w') as f:
            json.dump({'transcript': transcript}, f)
        print(f"Full transcript created: {full_transcript_path}")
    except Exception as e:
        print(f"Error in full_audio_transcription_pipeline: {e}")
        
def process_audio_files():
    directories = read_config(section="evaluations")
    config_params = read_config(section="config_params")
    try:
        base_path = directories['completedatasets']
        for n in os.listdir(base_path):
            initial_input_directory = os.path.join(base_path, n, 'originalvideos')
            audio_clip_output_dir = os.path.join(base_path, n, 'keyframe_audio_clips', 'whisper_audio_segments')
            full_audio_clip_output_dir = os.path.join(audio_clip_output_dir, 'full_whisper_segments')
            keyframe_dir = os.path.join(base_path, n, 'keyframes')
            keyframe_json_path = os.path.join(keyframe_dir, 'keyframe_data.json')
            keyframe_data = read_keyframe_data(keyframe_json_path)
            for audio_file in os.listdir(initial_input_directory):
                if audio_file.endswith('.m4a'):
                    audio_path = os.path.join(initial_input_directory, audio_file)
                    segment_audio_using_keyframes(audio_path, audio_clip_output_dir, keyframe_data, 5, suffix_='_keyframe')
                    individual_output_dir = os.path.join(audio_clip_output_dir, os.path.splitext(audio_file)[0])
                    if not os.path.exists(individual_output_dir):
                        os.makedirs(individual_output_dir)
                    convert_audio_files(initial_input_directory, individual_output_dir)
                    flac_file = os.path.splitext(audio_file)[0] + '.flac'
                    audio_path = os.path.join(individual_output_dir, flac_file)
                    audio_pipeline(audio_path, individual_output_dir, keyframe_data, 5)
            process_full_audio = string_to_bool(config_params.get("full_whisper_audio", "False"))
            if process_full_audio:
                if not os.path.exists(full_audio_clip_output_dir):
                    os.makedirs(full_audio_clip_output_dir)
                # Process the entire audio file for transcription
                full_audio_transcription_pipeline(audio_path, full_audio_clip_output_dir)
    except Exception as e:
        print(f"An exception occurred: {e}")