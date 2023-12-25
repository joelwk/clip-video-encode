import os
import json
import shutil
from pydub import AudioSegment
import os
import json
import shutil
from pydub import AudioSegment

def convert_audio_files(base_path, output_format="mp3"):
    for n in os.listdir(base_path):
        audio_clip_output_dir = os.path.join(base_path, n, 'keyframe_audio_clips', 'whisper_audio_segments')
        if not os.path.exists(audio_clip_output_dir):
            os.makedirs(audio_clip_output_dir)

        for subdir, dirs, files in os.walk(audio_clip_output_dir):
            for filename in files:
                file_path = os.path.join(subdir, filename)
                if filename.endswith(".flac"):
                    segment_info = filename.split('_')
                    if len(segment_info) >= 4 and segment_info[0] == 'video' and segment_info[2] == 'keyframe':
                        video_id = segment_info[1]
                        segment_idx = segment_info[3].split('.')[0]
                        output_filename = f"keyframe_audio_clip_{segment_idx}.{output_format}"
                        output_path = os.path.join(audio_clip_output_dir, output_filename)
                        if os.path.exists(output_path):
                            print(f"File {output_path} already exists. Overwriting.")
                        audio = AudioSegment.from_file(file_path, format="flac")
                        audio.export(output_path, format=output_format)
                        print(f"Converted {file_path} to {output_path}")
                        os.remove(file_path)
                        print(f"Removed {file_path}")
                elif filename.endswith(".json"):
                    new_json_path = os.path.join(audio_clip_output_dir, filename)
                    if file_path != new_json_path:
                        shutil.copy(file_path, new_json_path)
                        print(f"Copied {file_path} to {new_json_path}")
                    else:
                        print(f"File {new_json_path} is the same as the source. Skipping copy.")
                    with open(new_json_path, 'r', encoding='utf-8') as json_file:
                        try:
                            segments_data = json.load(json_file)
                        except json.JSONDecodeError:
                            print(f"Error reading JSON data from {new_json_path}")
                            continue
                        if filename == ("outputs.json"):
                            for segment_data in segments_data:
                                if isinstance(segment_data, dict) and "segment_idx" in segment_data:
                                    segment_idx = segment_data["segment_idx"]
                                    text_filename = f"keyframe_{segment_idx}.txt"
                                    text_path = os.path.join(audio_clip_output_dir, text_filename)
                                    with open(text_path, 'w', encoding='utf-8') as text_file: 
                                        text_file.write(segment_data.get("text", ""))
                                    print(f"Created text file for segment {segment_idx}")

def move_and_remove_subdirectory(audio_clip_output_dir):
    for subdir in os.listdir(audio_clip_output_dir):
        subdir_path = os.path.join(audio_clip_output_dir, subdir)
        if subdir.isdigit() and os.path.isdir(subdir_path) or subdir == 'full_whisper_segments' and os.path.isdir(subdir_path):
            try:
                shutil.rmtree(subdir_path)
            except Exception as e:
                print(f"Error removing {subdir_path}: {e}")
def main():
    base_path = './completedatasets'
    for n in os.listdir(base_path):
        audio_clip_output_dir = os.path.join(base_path, n ,'keyframe_audio_clips', 'whisper_audio_segments')
        convert_audio_files(base_path)  
        move_and_remove_subdirectory(audio_clip_output_dir)

if __name__ == '__main__':
    main()