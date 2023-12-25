import json
import os
import glob
import shutil
import numpy as np
import re

from evaluations.prepare import (
    model_clap, prepare_audio_labels,read_config, format_labels, softmax,get_all_video_ids,normalize_scores)

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def pair_and_classify_with_clap(audio_dir, json_dir, output_dir):
    params = read_config(section="evaluations")
    labels = read_config("labels")
    model = model_clap()
    multioutput_model, model_order_to_group_name, dfmetrics = prepare_audio_labels()
    os.makedirs(output_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(audio_dir, 'keyframe_*.json')))
    print('Found audio files:', audio_files)
    for audio_file in audio_files:
        base_name = os.path.basename(audio_file)
        print(f'Processing {base_name}')
        match = re.search(r'keyframe_(\d+)\.json', base_name)
        if match:
            keyframe_id = match.group(1)
            print(f'Match found: keyframe ID {keyframe_id}')
            json_path = os.path.join(json_dir, f"keyframe_{keyframe_id}.json")
            image_path = os.path.join(json_dir, f"keyframe_{keyframe_id}.png")
            if os.path.exists(json_path) and os.path.exists(image_path):
                json_file = json_path
                image_file = image_path
                emotions_list = format_labels(labels, 'emotions')
                text_features = model.get_text_embedding(emotions_list, use_tensor=False)
                text_features = normalize_vectors(text_features)
                audio_embed = np.squeeze(model.get_audio_embedding_from_filelist([audio_file], use_tensor=False))
                audio_embed_normalized = normalize_vectors(audio_embed.reshape(1, -1))
                similarity_scores = np.abs(audio_embed_normalized @ text_features.T)
                similarity_probs = softmax(float(params['scalingfactor']) * audio_embed_normalized @ text_features.T)
                similarity_probs = similarity_probs.tolist()
                sorted_emotion_score_pairs = {k: v for k, v in sorted({format_labels(labels, 'emotions')[i]: float(similarity_scores[0][i]) for i in range(len(format_labels(labels, 'emotions')))}.items(), key=lambda item: item[1], reverse=True)}
                sorted_emotion_probs_pairs = {k: v for k, v in sorted({format_labels(labels, 'emotions')[i]: float(similarity_probs[0][i]) for i in range(len(format_labels(labels, 'emotions')))}.items(), key=lambda item: item[1], reverse=True)}
                json_data = {
                    "audio_file_name": base_name,
                    "sorted_emotion_prob_pairs": sorted_emotion_probs_pairs,
                    "sorted_emotion_score_pairs":sorted_emotion_score_pairs}
                output_json_path = f"{output_dir}/{base_name.replace('.mp3', '')}_analysis.json"
                with open(output_json_path, 'w') as out_f:
                    json.dump(json_data, out_f, indent=4)
                np.save(os.path.join(output_dir, base_name.replace('.mp3', '') + '_analysis.npy'), audio_embed)
            else:
                print(f"Expected JSON file: {json_path}")
                print(f"Expected Image file: {image_path}")
                print("Either the JSON or the Image file or both were not found. Please check the paths and file names.")

def combine_emotion_scores(image_json_path, audio_json_path, output_path):
    # Load JSON data
    with open(image_json_path, 'r') as file:
        image_data = json.load(file)
    with open(audio_json_path, 'r') as file:
        audio_data = json.load(file)
    combined_emotions = {}
    # Combine image emotions
    for emotion, prob in image_data['emotions_scores'].items():
        combined_emotions[emotion] = {'image': prob}
    # Combine audio emotions
    for emotion, score in audio_data['sorted_emotion_score_pairs'].items():
        if emotion in combined_emotions:
            combined_emotions[emotion]['audio'] = score
        else:
            combined_emotions[emotion] = {'audio': score}
    sorted_combined_emotions = dict(sorted(combined_emotions.items(), key=lambda item: -sum(filter(None, item[1].values()))))
    output = {
        "image name": os.path.basename(image_json_path).replace('.json', '.png'),
        "emotions": sorted_combined_emotions
    }
    with open(output_path, 'w') as file:
        json.dump(output, file, indent=4)
        
def process_all_keyframes(video_base_path, audio_processed_base_path, output_base_path):
    evaluations_base_path = os.path.dirname(audio_processed_base_path)
    path_image_audio_pairs = os.path.join(evaluations_base_path, "image_audio_pairs")
    for video_dir in glob.glob(video_base_path + '/*'):
        video_id = os.path.basename(video_dir)
        path_image_audio_pairs = os.path.join(evaluations_base_path,"image_audio_pairs", video_id)
        video_output_dir = os.path.join(output_base_path, f'{video_id}')
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        for image_json_file in glob.glob(video_dir + '/*.json'):
            match = re.search(r'segment_(\d+)\.json', os.path.basename(image_json_file))
            if match:
                keyframe_id = match.group(1)
                print(f'Processing keyframe {keyframe_id} with video {video_id}')
                audio_json_path = os.path.join(audio_processed_base_path, f'{video_id}', f'keyframe_{keyframe_id}_analysis.json')
                audio_json_path_vocals = os.path.join(audio_processed_base_path, f'{video_id}', f'keyframe_{keyframe_id}_vocals_analysis.json')
                audio_json_to_use = audio_json_path_vocals if os.path.exists(audio_json_path_vocals) else audio_json_path
                text_filename = f"video_{video_id}_keyframe_audio_clip_{keyframe_id}.txt"
                text_file_path = os.path.join(path_image_audio_pairs, text_filename)
                if os.path.exists(audio_json_to_use):
                    output_json_path = os.path.join(video_output_dir, f'output_combined_emotions_{keyframe_id}.json')
                    combine_emotion_scores(image_json_file, audio_json_to_use, output_json_path)
                    print(f'Combined JSON created for keyframe {keyframe_id} video {video_id}')
                    keyframe_base = os.path.splitext(image_json_file)[0]  # Remove .json extension
                    for image_file in glob.glob(f"{keyframe_base}*.png"):
                        shutil.copy(image_file, video_output_dir)
                        print(f"Processed image file copied: {image_file}")
                    for npy_file in glob.glob(f"{keyframe_base}*_image_features.npy"):
                        shutil.copy(npy_file, video_output_dir)
                        print(f"Processed image NPY file copied: {npy_file}")
                    audio_file_path = audio_json_to_use.replace('_analysis.json', '.mp3')
                    audio_npy_file_path = audio_json_to_use.replace('_analysis.json', '_audio_features.npy')
                    if os.path.exists(audio_file_path):
                        shutil.copy(audio_file_path, video_output_dir)
                        shutil.copy(audio_npy_file_path, video_output_dir)
                        shutil.copy(text_file_path, video_output_dir)
                        print(f'Processed audio file copied for keyframe {keyframe_id} of video {video_id}')

def main():
    params = read_config(section="evaluations")
    video_ids = get_all_video_ids(params['completedatasets'])
    for video in video_ids:
        audio_directory = f"./evaluations/audio_evaluations/{str(video)}"
        json_image_directory = f"./evaluations/image_evaluations/{str(video)}"
        paired_evaluations = f"./evaluations/paired_evaluations/"
        all_image = f"./evaluations/image_evaluations"
        all_audio = f"./evaluations/audio_evaluations"
        image_audio_pairs = f"./evaluations/audio_evaluations"
        pair_and_classify_with_clap(audio_directory, json_image_directory, audio_directory)
        process_all_keyframes(all_image, all_audio, paired_evaluations)

if __name__ == "__main__":
    main()