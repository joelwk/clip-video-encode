import json
import os
import glob
import shutil

from evaluations.prepare import (
    model_clap,prepare_audio_labels
)

def pair_and_classify_with_clap(audio_dir, json_dir, output_dir, model_checkpoint):
    params = read_config(section="evaluations")
    model = model_clap()
    multioutput_model, model_order_to_group_name, dfmetrics = prepare_audio_labels()
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Iterate over audio files
    audio_files = sorted(glob.glob(audio_dir + '/*_vocals.mp3', recursive=True))
    for audio_file in audio_files:
        base_name = os.path.basename(audio_file)
        print(f'Processing {base_name}')
        # Regular expression to extract the keyframe index
        match = re.search(r'segment_(\d+)', base_name)
        if match:
            image_id = match.group(1)
            print(image_id)
            # Construct filenames for JSON and PNG files
            json_pattern = f"{json_dir}/keyframe_{image_id}*.json"
            image_pattern = f"{json_dir}/keyframe_{image_id}*.png"
            # Find the first file that matches the pattern
            json_files = glob.glob(json_pattern)
            image_files = glob.glob(image_pattern)
            if json_files and image_files:
                json_file = json_files[0]
                image_file = image_files[0]
                text_features_path = os.path.join(params['embeddings'], 'text_features.npy')
                text_features = np.load(text_features_path)
                audio_embed = np.squeeze(model.get_audio_embedding_from_filelist([audio_file], use_tensor=False))
                # Get and normalize text embeddings for emotions
                text_embed = model.get_text_embedding(text_features, use_tensor=False)
                # Calculate similarity scores
                similarity_scores = softmax(100 * audio_embed.reshape(1, -1) @ text_embed.T)
                # Convert similarity scores from NumPy array to list
                similarity_scores = similarity_scores.tolist()
                sorted_emotion_score_pairs = {k: v for k, v in sorted({emotions[i]: float(similarity_scores[0][i]) for i in range(len(emotions))}.items(), key=lambda item: item[1], reverse=True)}
                # Prepare and save JSON data with classification results
                json_data = {
                    "audio_file_name": base_name,
                    "sorted_emotion_score_pairs": sorted_emotion_score_pairs,
                }
                output_json_path = f"{output_dir}/{base_name.replace('.mp3', '')}_analysis.json"
                with open(output_json_path, 'w') as out_f:
                    json.dump(json_data, out_f, indent=4)
                # Save the image features as .npy files for further analysis
                np.save(os.path.join(output_dir, base_name.replace('.mp3', '') + '_analysis.npy'), audio_embed)

def combine_emotion_scores(image_json_path, audio_json_path, output_path):
    # Load JSON data
    with open(image_json_path, 'r') as file:
        image_data = json.load(file)
    with open(audio_json_path, 'r') as file:
        audio_data = json.load(file)
    combined_emotions = {}
    # Combine image emotions
    for emotion, score in image_data['emotions'].items():
        combined_emotions[emotion] = {'image': score}
    # Combine audio emotions
    for emotion, score in audio_data['sorted_emotion_score_pairs'].items():
        if emotion in combined_emotions:
            combined_emotions[emotion]['audio'] = score
        else:
            combined_emotions[emotion] = {'audio': score}
    # Sort emotions by total score, descending, non-zero scores at the top
    sorted_combined_emotions = dict(sorted(combined_emotions.items(), key=lambda item: -sum(filter(None, item[1].values()))))
    # Prepare final output
    output = {
        "image name": os.path.basename(image_json_path).replace('.json', '.png'),
        "emotions": sorted_combined_emotions
    }
    print(output_path)
    # Save to output JSON file
    with open(output_path, 'w') as file:
        json.dump(output, file, indent=4)

def process_all_keyframes(video_base_path, audio_processed_base_path, vocals_base_path, output_base_path):
    # Iterate through all video directories
    for video_dir in glob.glob(video_base_path + '/video_*'):
        video_id = os.path.basename(video_dir).split('_')[1]
        # Create a directory for the current video in the output base path
        video_output_dir = os.path.join(output_base_path, f'video_{video_id}')
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        # Iterate through all keyframe JSON files in the video directory
        for image_json_file in glob.glob(video_dir + '/keyframe_*.json'):
            keyframe_id = os.path.basename(image_json_file).split('_')[1]
            print(f'Processing keyframe {keyframe_id} of video {video_id}')
            # Construct paths for corresponding audio and vocals JSON files
            audio_json_path = os.path.join(audio_processed_base_path, f'video_{video_id}', f'segment_{keyframe_id}__keyframe_vocals_analysis.json')
            # Check if both audio and vocals JSON files exist
            if os.path.exists(audio_json_path):
                output_json_path = os.path.join(video_output_dir, f'output_combined_emotions_{keyframe_id}.json')
                combine_emotion_scores(image_json_file, audio_json_path, output_json_path)
                print(f'Combined JSON created for keyframe {keyframe_id} of video {video_id}')
                # Copy the image file to the output directory
                image_file_path = image_json_file.replace('.json', '.png')
                if os.path.exists(image_file_path):
                    shutil.copy(image_file_path, video_output_dir)
                # Copy the audio file from the processed folder to the output directory
                # join audio path still
                audio_file_path = audio_json_path.replace('_analysis.json', '.mp3')
                if os.path.exists(audio_file_path):
                    shutil.copy(audio_file_path, video_output_dir)
                audio_file_path_vocals = audio_json_path.replace('_analysis.json', '_vocals.mp3')
                if os.path.exists(audio_file_path):
                    shutil.copy(audio_file_path, video_output_dir)
                    print(f'Processed audio file copied for keyframe {keyframe_id} of video {video_id}')
