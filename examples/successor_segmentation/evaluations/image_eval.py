from PIL import Image
import os
import json
import subprocess
import argparse
import numpy as np
import glob
from evaluations.prepare import read_config, generate_embeddings,tensor_to_array,display_image_from_file,remove_duplicate_extension,print_top_n,softmax,normalize_scores,process_keyframe_audio_pairs, get_embeddings, model
import cv2
import open_clip
import torch

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
        
def zeroshot_classifier(image_path, video_identifier, output_dir, display_image=True):
    params = read_config(section="evaluations")
    model_clip, preprocess_train, preprocess_val, tokenizer = model()
    text_features, text_features_if_person, text_features_type_person, text_features_if_number_of_faces, text_features_orientation, text_features_if_engaged, text_features_valence = get_embeddings(model_clip, tokenizer)

    # Form the paths to the embeddings
    text_features_path = os.path.join(params['labels'], 'text_features.npy')
    text_features_if_person_path = os.path.join(params['labels'], 'text_features_if_person.npy')
    text_features_type_person_path = os.path.join(params['labels'], 'text_features_type_person.npy')
    text_features_if_number_of_faces_path = os.path.join(params['labels'], 'text_features_number_of_faces.npy')
    text_features_orientation_path = os.path.join(params['labels'], 'text_features_orientation.npy')
    text_features_if_engaged_path = os.path.join(params['labels'], 'text_features_if_engaged.npy')
    text_features_valence_path = os.path.join(params['labels'], 'text_valence.npy')

    # Load embeddings
    text_features = np.load(text_features_path)
    text_features_if_person = np.load(text_features_if_person_path)
    text_features_type_person = np.load(text_features_type_person_path)
    text_features_if_number_of_faces = np.load(text_features_if_number_of_faces_path)
    text_features_orientation = np.load(text_features_orientation_path)
    text_features_if_engaged = np.load(text_features_if_engaged_path)
    text_features_valence = np.load(text_features_valence_path)
  

    labels = read_config(section="labels")
    
    # Set up the output directory for processed images
    run_output_dir = os.path.join(output_dir, video_identifier)
    os.makedirs(run_output_dir, exist_ok=True)

    # Load and preprocess the image
    img = Image.open(image_path)
    image_preprocessed = preprocess_val(img).unsqueeze(0)

    # Encode the image using the CLIP model and normalize the features
    image_features = model_clip.encode_image(image_preprocessed)
    image_features = image_features.detach().numpy()
    image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)

    # Calculate probabilities for different categories using softmax
    is_person_probs = softmax(100.0 * normalize_scores(image_features @ text_features_if_person.T))
    type_person_probs = softmax(100.0 * normalize_scores(image_features @ text_features_type_person.T))
    face_probs = softmax(100.0 * normalize_scores(image_features @ text_features_if_number_of_faces.T))
    orientation_probs = softmax(100.0 * normalize_scores(image_features @ text_features_orientation.T))
    engagement_probs = softmax(100.0 * normalize_scores(image_features @ text_features_if_engaged.T))
    text_probs_emotions = softmax(100.0 * normalize_scores(image_features @ text_features.T))
    text_probs_valence = softmax(100.0 * normalize_scores(image_features @ text_features_valence.T))

    # Determine if the image is a close-up of a face
    face_detected = is_good_image(is_person_probs[0],face_probs[0], orientation_probs[0], engagement_probs[0])
    # Process the image if a face is detected
    if face_detected:
        # Optionally display the image
        if display_image:
            display_image_from_file(image_path)
            # Print top probabilities for different categories
            print_top_n(is_person_probs[0], labels['check_if_person'])
            print_top_n(type_person_probs[0], labels['check_type_person'])
            print_top_n(face_probs[0], labels['number_of_faces'])
            print_top_n(orientation_probs[0], labels['orientation_labels'])
            print_top_n(engagement_probs[0], labels['engagement_labels'])
            print_top_n(text_probs_emotions[0], labels['emotions'])
            print_top_n(text_probs_valence[0], labels['valence'])

        # Save the processed image
        filename = os.path.basename(image_path)
        filename_without_ext = filename.split('.')[0]
        filename = remove_duplicate_extension(filename)
        save_path = os.path.join(run_output_dir, filename)
        img.save(save_path)

        # Sort and store the detection scores for faces, emotions, and valence
        sorted_face_detection_scores = {k: v for k, v in sorted({labels['check_if_person'][i]: float(is_person_probs[0][i]) for i in range(len(labels['check_if_person']))}.items(), key=lambda item: item[1], reverse=True)}
        sorted_emotions = {k: v for k, v in sorted({labels['emotions'][i]: float(text_probs_emotions[0][i]) for i in range(len(labels['emotions']))}.items(), key=lambda item: item[1], reverse=True)}
        sorted_valence = {k: v for k, v in sorted({labels['valence'][i]: float(text_probs_valence[0][i]) for i in range(len(labels['valence']))}.items(), key=lambda item: item[1], reverse=True)}

        # Convert NumPy boolean to Python boolean for JSON serialization
        face_detected_python_bool = bool(face_detected)

        # Prepare and save JSON data with classification results
        json_data = {
            "image_path": filename,
            "face_detected": face_detected_python_bool,
            "face_detection_scores": sorted_face_detection_scores,
            "emotions": sorted_emotions,
            "valence": sorted_valence
        }
        json_filename = filename_without_ext + '.json'
        with open(os.path.join(run_output_dir, json_filename), 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        # Save the image features as .npy files for further analysis
        npy_filename_base = filename_without_ext
        np.save(os.path.join(run_output_dir, npy_filename_base + '_image_features.npy'), image_features)

def load_key_image_files(vid, params):
    # Returns an iterator over sorted keyframe image files for a given video ID
    pattern = os.path.join(params['completedatasets'], str(vid), "keyframes", "*.png")
    return iter(sorted(glob.glob(pattern)))

def load_key_audio_files(vid, params):
    # Returns an iterator over sorted keyframe audio files for a given video ID
    pattern = os.path.join(params['completedatasets'], str(vid), "keyframe_audio_clips", "whisper_audio_segments", "*.mp3")
    return iter(sorted(glob.glob(pattern)))

def get_all_video_ids(directory):
    # Returns an iterator over video IDs in the given directory
    return iter([int(os.path.basename(f)) for f in glob.glob(os.path.join(directory, '*'))])

def main():
    # Read configurations
    params = read_config(section="evaluations")
    video_ids = get_all_video_ids(params['completedatasets'])
    for video in video_ids:
        try:
            keyframes = load_key_image_files(video, params)
            audios = load_key_audio_files(video, params)
            for keyframe in keyframes:
                zeroshot_classifier(keyframe, str(video), params['outputs'], display_image=True)

            # Directories as defined in config.ini
            image_dir = os.path.join(params['outputs'], "image_evaluations", str(video))
            output_dir = os.path.join(params['outputs'], "image_audio_pairs", str(video))
            audio_dir = os.path.join(params['completedatasets'], str(video), "keyframe_audio_clips", "whisper_audio_segments")
            
            # Process keyframe-audio pairs
            process_keyframe_audio_pairs(image_dir, audio_dir, output_dir)
        except Exception as e:
            print(f"Failed to process images and pair with audio for {video}: {e}")

if __name__ == '__main__':
    main()