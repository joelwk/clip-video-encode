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

''' 
segment_video - segment video by time stamps to output individual mp4's
segment_video - segment audio by time stamps to output individual m4a's and flac's with associated whisper transcripts
compute_embeddings - average each embedding value for each video and output to a single npy file - disable if segment_video is False ** need to condition this in the future*
specific_videos - indicate specific video with [1,2,3] for all 3 videos, or [1] for 1 video

'''

def run_all_scripts(segment_video=False, segment_audio=True, compute_embeddings=False, specific_videos=None):

    # Run the main function from rename_and_move.py
    rename_and_move_main()
    
    # Run the main function from successor_segmentation and fold_seams
    run_analysis(SegmentSuccessorAnalyzer)

    fold_seams_main(segment_video, segment_audio, specific_videos)
    if compute_embeddings:
        segment_averaging_main()

    # Run the main function from move_and_group.py
    move_and_group_main()
    process_audio_files()
    convert_types_main()
    # Run the main function from save_to_webdataset.py
    save_to_webdataset_main()

if __name__ == "__main__":
    run_all_scripts()