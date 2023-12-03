# Import the main functions from your other Python scripts
from srcs.rename_and_move import main as rename_and_move_main
from srcs.segment_averaging import main as segment_averaging_main
from srcs.move_and_group import main as move_and_group_main
from srcs.save_to_webdataset import main as save_to_webdataset_main
from srcs.whisper import process_audio_files
# Import and run your analysis from SegmentSuccessorAnalyzer and fold_seams main function
from srcs.successor_segmentation import SegmentSuccessorAnalyzer, run_analysis
from srcs.fold_seams import main as fold_seams_main
from srcs.convert_types import main as convert_types_main
import configparser
from srcs.load_data import read_config, string_to_bool

''' 
segment_video - segment video by time stamps to output individual mp4's
segment_video - segment audio by time stamps to output individual m4a's and flac's with associated whisper transcripts
compute_embeddings - average each embedding value for each video and output to a single npy file - disable if segment_video is False ** need to condition this in the future*
specific_videos - indicate specific video with [1,2,3] for all 3 videos, or [1] for 1 video

'''

def run_all_scripts(config):
    segment_video = string_to_bool(config.get("segment_video", "False"))
    segment_audio = string_to_bool(config.get("segment_audio", "True"))
    compute_embeddings = string_to_bool(config.get("compute_embeddings", "False"))
    specific_videos_str = config.get("specific_videos", "")
    if specific_videos_str and specific_videos_str != "None":
        specific_videos = [int(x.strip()) for x in specific_videos_str.strip('[]').split(',')]
    else:
        specific_videos = None
    rename_and_move_main()
    run_analysis(SegmentSuccessorAnalyzer)
    fold_seams_main(segment_video, segment_audio, specific_videos)

    if compute_embeddings:
        segment_averaging_main()

    move_and_group_main()
    process_audio_files()
    convert_types_main()
    save_to_webdataset_main()

def initialize_and_run():
    config = read_config(section="config_params")
    run_all_scripts(config)

if __name__ == "__main__":
    initialize_and_run()