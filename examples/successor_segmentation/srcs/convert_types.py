import os
import json
import shutil
from pydub import AudioSegment

def convert_audio_files(output_format="mp3"):
    base_path = './completedatasets/'
    for n in os.listdir(base_path):
        audio_clip_output_dir = os.path.join(base_path, n, 'keyframe_audio_clips', 'whisper_audio_segments')
    if not os.path.exists(audio_clip_output_dir):
        os.makedirs(audio_clip_output_dir)
    # Traverse subdirectories
    for subdir, dirs, files in os.walk(audio_clip_output_dir):
        for filename in files:
            if filename.endswith(".flac"):
                flac_path = os.path.join(subdir, filename)
                # Extract the base filename without the format and additional suffix
                base_filename = filename.replace("_whisper.flac", "")

                # Extract the numeric part for segment matching
                digits = ''.join(filter(str.isdigit, base_filename))
                if digits:
                    segment_idx = int(digits)
                else:
                    continue  # Skip this file if no digits found

                output_filename = f"keyframe_audio_clip_{segment_idx}.{output_format}"
                output_path = os.path.join(audio_clip_output_dir, output_filename)

                if os.path.exists(output_path):
                    print(f"File {output_path} already exists. Overwriting.")

                audio = AudioSegment.from_file(flac_path, format="flac")
                audio.export(output_path, format=output_format)
                print(f"Converted {flac_path} to {output_path}")

                # Remove the original flac file
                os.remove(flac_path)
                print(f"Removed {flac_path}")

            elif filename.endswith(".json"):
                # Handle JSON files
                json_path = os.path.join(subdir, filename)
                new_json_path = os.path.join(audio_clip_output_dir, filename)  # Keep original filename
                shutil.copy(json_path, new_json_path)
                print(f"Copied {json_path} to {new_json_path}")

                # Read JSON data for text file creation
                with open(new_json_path, 'r') as json_file:
                    try:
                        segments_data = json.load(json_file)
                    except json.JSONDecodeError:
                        print(f"Error reading JSON data from {new_json_path}")
                        continue

                # Process each segment data
                for segment_data in segments_data:
                    if isinstance(segment_data, dict) and "segment_idx" in segment_data:
                        segment_idx = segment_data["segment_idx"]
                        text_filename = f"keyframe_audio_clip_{segment_idx}.txt"
                        text_path = os.path.join(audio_clip_output_dir, text_filename)
                        with open(text_path, 'w') as text_file:
                            text_file.write(segment_data.get("text", ""))
                        print(f"Created text file for segment {segment_idx}")
                        
    # Iterate over subdirectories in the parent directory
    for subdir in os.listdir(audio_clip_output_dir):
        subdir_path = os.path.join(audio_clip_output_dir, subdir)
        if os.path.isdir(subdir_path) and subdir.isdigit():
            shutil.rmtree(subdir_path)
            print(f"Removed subdirectory {subdir_path}")
                     
def main():
    convert_audio_files()
if __name__ == '__main__':
    main()
