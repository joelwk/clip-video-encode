import configparser
import shutil
import sys
import os
import json

import numpy as np
import open_clip
import tensorflow as tf
import torch
from PIL import Image

def read_config(section, config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    config = configparser.ConfigParser()
    config.read(config_path)
    if section not in config.sections():
        raise KeyError(f"Section {section} not found in configuration file.")
    return {key: config[section][key] for key in config[section]}

# Function to convert tensor to array
def tensor_to_array(tensor):
    return tensor.cpu().numpy()

# Function to tokenize and generate embeddings
def generate_embeddings(tokenizer, model_clip, prompts, file_name):
    if not os.path.exists(file_name + '.npy'):
        text = tokenizer(prompts)
        with torch.no_grad():
            text_features = model_clip.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = tensor_to_array(text_features)
        np.save(file_name, text_features)
    else:
        text_features = np.load(file_name + '.npy')
    return text_features

# Function to save dictionaries as JSON
def save_dict_as_json(dictionary, file_name):
    if not os.path.exists(file_name + '.json'):
        with open(file_name + '.json', 'w') as file:
            json.dump(dictionary, file, indent=4)

def display_image_from_file(image_path):
    img = Image.open(image_path)
    display(img)

def print_top_n(probs, labels):
    top_n_indices = np.argsort(probs)[::-1][:5]
    for i in top_n_indices:
        print(f"{labels[i]}: {probs[i]:.4f}")

def remove_duplicate_extension(filename):
    parts = filename.split('.')
    if len(parts) > 2 and parts[-1] == parts[-2]:
        return '.'.join(parts[:-1])
    return filename

def normalize_scores(scores):
    mean = np.mean(scores, axis=1, keepdims=True)
    std = np.std(scores, axis=1, keepdims=True)
    normalized_scores = (scores - mean) / std
    return normalized_scores

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def read_labels(section, config_path):
    config = read_config(section, config_path)
    labels = config['labels'].split(', ')
    return labels

def process_keyframe_audio_pairs(faces_dir, audio_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all keyframe image filenames from the faces directory
    keyframe_filenames = [f for f in os.listdir(faces_dir) if f.endswith('.png')]

    # Process each keyframe image file
    for keyframe_filename in keyframe_filenames:
        digit_str = ''.join(filter(str.isdigit, keyframe_filename))

        if digit_str:
            segment_idx = int(digit_str)
            # Construct the corresponding audio filename
            audio_filename = f"keyframe_audio_clip_{segment_idx}.mp3"
            audio_path = os.path.join(audio_dir, audio_filename)
            # Construct the corresponding text filename
            text_filename = f"keyframe_audio_clip_{segment_idx}.txt"
            text_path = os.path.join(audio_dir, text_filename)
            # Check if the corresponding audio file exists
            if os.path.isfile(audio_path):
                # Copy the audio file to the output directory
                output_audio_path = os.path.join(output_dir, remove_duplicate_extension(audio_filename))
                shutil.copy(audio_path, output_audio_path)
                print(f"Copied {audio_path} to {output_audio_path}")
            if os.path.isfile(text_path):
                # Copy the text file to the output directory
                output_text_path = os.path.join(output_dir, remove_duplicate_extension(text_filename))
                shutil.copy(text_path, output_text_path)
                print(f"Copied {text_path} to {output_text_path}")
        else:
            print(f"No digits found in filename: {keyframe_filename}")

def model(config_path):
    model_config = read_config('evaluations', config_path)
    model_name = model_config.get('model')
    model_clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model_clip, preprocess_train, preprocess_val, tokenizer

def get_embeddings(model_clip, tokenizer, config_path):
    evals = read_config('evaluations', config_path)
    emotions = read_labels('emotions', config_path)
    check_if_person = read_labels('checkifperson', config_path)
    check_type_person = read_labels('checktypeperson', config_path)
    number_of_faces = read_labels('numberoffaces', config_path)
    engagement_labels = read_labels('engagementlabels', config_path)
    orientation_labels = read_labels('orientationlabels', config_path)
    valence = read_labels('valence', config_path)
    text_features = generate_embeddings(tokenizer, model_clip, emotions, f"{evals['labels']}/text_features.npy")
    text_features_if_person = generate_embeddings(tokenizer, model_clip, check_if_person, f"{evals['labels']}/text_features_if_person.npy")
    text_features_type_person = generate_embeddings(tokenizer, model_clip, check_type_person, f"{evals['labels']}/text_features_type_person.npy")
    text_features_if_number_of_faces = generate_embeddings(tokenizer, model_clip, number_of_faces, f"{evals['labels']}/text_features_number_of_faces.npy")
    text_features_orientation = generate_embeddings(tokenizer, model_clip, orientation_labels, f"{evals['labels']}/text_features_orientation.npy")
    text_features_if_engaged = generate_embeddings(tokenizer, model_clip, engagement_labels, f"{evals['labels']}/text_features_if_engaged.npy")
    text_features_valence = generate_embeddings(tokenizer, model_clip, valence, f"{evals['labels']}/text_valence.npy")
    return text_features, text_features_if_person, text_features_type_person, text_features_if_number_of_faces, text_features_orientation, text_features_if_engaged, text_features_valence

def main():
    try:
        config_path = './clip-video-encode/examples/successor_segmentation/config.ini'
        # Load the model and tokenizer
        model_clip, preprocess_train, preprocess_val, tokenizer = model(config_path)
        # Generate and return embeddings
        embeddings = get_embeddings(model_clip, tokenizer, config_path)
        return model_clip, preprocess_train, preprocess_val, tokenizer, *embeddings
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        sys.exit(1)
if __name__ == "__main__":
    main()