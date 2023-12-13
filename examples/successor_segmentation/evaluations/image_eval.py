from PIL import Image
import os
import shutil
import json
import subprocess
import argparse
import numpy as np
import glob
import torch
from evaluations.prepare import (
    read_config, generate_embeddings, format_labels,
    remove_duplicate_extension, process_keyframe_audio_pairs, get_embeddings, model_clip, 
    display_image_from_file, print_top_n, normalize_scores, softmax, sort_and_store_scores,load_key_image_files, load_key_audio_files, get_all_video_ids
)
    
def is_good_image(is_person, face_probs, orientation_probs, engagement_probs):
    # Define thresholds
    is_person_threshold = 0.95  # High probability of the subject being a person
    single_face_threshold = 0.95  # High probability of there being only one face
    facing_forward_threshold = 0.95  # High probability of the subject facing forward
    engagement_threshold = 0.95  # High probability of the subject looking at the camera or not, depending on preference
    type_person_threshold = 0.5  # Threshold for type of person detection

    # Check conditions
    is_person_detected = is_person[1] > is_person_threshold
    single_face_detected = face_probs[0] > single_face_threshold
    facing_forward = orientation_probs[0] > facing_forward_threshold
    engaged = engagement_probs[0] > engagement_threshold or engagement_probs[1] > engagement_threshold
    # Return True if the image meets the criteria for being "Good"
    return is_person_detected and single_face_detected and facing_forward and engaged

def zeroshot_classifier(image_path, video_identifier, output_dir, display_image=False):
    params = read_config(section="evaluations")
    labels = read_config("labels")
    model, preprocess_train, preprocess_val, tokenizer = model_clip()
    get_embeddings(model, tokenizer)

    # Form the paths to the embeddings
    text_features_path = os.path.join(params['embeddings'], 'text_features.npy')
    text_features_if_person_path = os.path.join(params['embeddings'], 'text_features_if_person.npy')
    text_features_type_person_path = os.path.join(params['embeddings'], 'text_features_type_person.npy')
    text_features_if_number_of_faces_path = os.path.join(params['embeddings'], 'text_features_number_of_faces.npy')
    text_features_orientation_path = os.path.join(params['embeddings'], 'text_features_orientation.npy')
    text_features_if_engaged_path = os.path.join(params['embeddings'], 'text_features_if_engaged.npy')
    text_features_valence_path = os.path.join(params['embeddings'], 'text_valence.npy')

    # Load embeddings
    text_features = np.load(text_features_path)
    text_features_if_person = np.load(text_features_if_person_path)
    text_features_type_person = np.load(text_features_type_person_path)
    text_features_if_number_of_faces = np.load(text_features_if_number_of_faces_path)
    text_features_orientation = np.load(text_features_orientation_path)
    text_features_if_engaged = np.load(text_features_if_engaged_path)
    text_features_valence = np.load(text_features_valence_path)
    
    # Set up the output directory for processed images
    run_output_dir = os.path.join(output_dir, video_identifier)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Load and preprocess the image
    img = Image.open(image_path)
    image_preprocessed = preprocess_val(img).unsqueeze(0)

    # Encode the image using the CLIP model and normalize the features
    image_preprocessed = image_preprocessed.to('cuda' if torch.cuda.is_available() else 'cpu')
    image_features = model.encode_image(image_preprocessed)
    image_features = image_features.detach().cpu().numpy()
    image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
    
    # Calculate probabilities for different categories using softma
    is_person_probs = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features_if_person.T))
    type_person_probs = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features_type_person.T))
    face_probs = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features_if_number_of_faces.T))
    orientation_probs = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features_orientation.T))
    engagement_probs = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features_if_engaged.T))
    text_probs_emotions = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features.T))
    text_score_emotions = image_features @ text_features.T
    text_probs_valence = softmax(float(params['scalingfactor']) * image_features @ text_features_valence.T)
    face_detected = is_good_image(is_person_probs[0], face_probs[0], orientation_probs[0], engagement_probs[0])

    if face_detected:
        if display_image:
            display_image_from_file(image_path)
            print_top_n(is_person_probs[0], format_labels(labels, 'checkifperson'))
            print_top_n(type_person_probs[0], format_labels(labels, 'checktypeperson'))
            print_top_n(face_probs[0], format_labels(labels, 'numberoffaces'))
            print_top_n(orientation_probs[0], format_labels(labels, 'orientationlabels'))
            print_top_n(engagement_probs[0], format_labels(labels, 'engagementlabels'))
            print_top_n(text_probs_emotions[0], format_labels(labels, 'emotions'))
            print_top_n(text_probs_valence[0], format_labels(labels, 'valence'))
            
        filename = os.path.basename(image_path)
        filename_without_ext = filename.split('.')[0]
        filename = remove_duplicate_extension(filename)
        save_path = os.path.join(run_output_dir, filename)
        img.save(save_path)
        sorted_face_detection_scores = sort_and_store_scores(is_person_probs[0], format_labels(labels, 'checktypeperson'))
        sorted_emotions = sort_and_store_scores(text_probs_emotions[0], format_labels(labels, 'emotions'))
        sorted_emotions_scores = sort_and_store_scores(text_score_emotions[0], format_labels(labels, 'emotions'))
        sorted_valence = sort_and_store_scores(text_probs_valence[0], format_labels(labels, 'valence'))
        face_detected_python_bool = bool(face_detected)
        json_data = {
            "image_path": filename,
            "face_detected": face_detected_python_bool,
            "face_detection_scores": sorted_face_detection_scores,
            "emotions_probs": sorted_emotions,
            "emotions_scores":sorted_emotions_scores,
            "valence": sorted_valence}
        json_filename = filename_without_ext + '.json'
        with open(os.path.join(run_output_dir, json_filename), 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        npy_filename_base = filename_without_ext
        np.save(os.path.join(run_output_dir, npy_filename_base + '_image_features.npy'), image_features)
        return True 
    return False 

def main():
    params = read_config(section="evaluations")
    video_ids = get_all_video_ids(params['completedatasets'])
    for video in video_ids:
        try:
            face_detected_in_video = False
            keyframes = load_key_image_files(video, params)
            for keyframe in keyframes:
                if zeroshot_classifier(keyframe, str(video), os.path.join(params['outputs'], "image_evaluations"), display_image=False):
                    face_detected_in_video = True
            if not face_detected_in_video:
                video_dir = os.path.join(params['outputs'], "image_evaluations", str(video))
                if os.path.exists(video_dir):
                    shutil.rmtree(video_dir)
                    print(f"No faces detected in any keyframes of video {video}. Directory {video_dir} removed.")
                continue
            image_dir = os.path.join(params['outputs'], "image_evaluations", str(video))
            output_dir = os.path.join(params['outputs'], "image_audio_pairs", str(video))
            audio_dir = os.path.join(params['completedatasets'], str(video), "keyframe_audio_clips", "whisper_audio_segments")
            process_keyframe_audio_pairs(image_dir, audio_dir, output_dir)
        except Exception as e:
            print(f"Failed to process images and pair with audio for video {video}: {e}")

if __name__ == '__main__':
    main()
