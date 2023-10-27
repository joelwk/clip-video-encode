# Import the main functions from your other Python scripts
from srcs.rename_and_move import main as rename_and_move_main
from srcs.segment_averaging import main as segment_averaging_main
from srcs.move_and_group import main as move_and_group_main
from srcs.save_to_webdataset import main as save_to_webdataset_main
from srcs.whisper import process_audio_files
# Import and run your analysis from SegmentSuccessorAnalyzer and fold_seams main function
from srcs.successor_segmentation import SegmentSuccessorAnalyzer, run_analysis
from srcs.fold_seams import main as fold_seams_main

def run_all_scripts():
    # Run the main function from rename_and_move.py
    rename_and_move_main()
    
    # Run the main function from successor_segmentation and fold_seams
    run_analysis(SegmentSuccessorAnalyzer)
    fold_seams_main()
    
    # Run the main function from segment_averaging.py
    segment_averaging_main()
  
    # Run the main function from move_and_group.py
    move_and_group_main()
    process_audio_files()
    # Run the main function from save_to_webdataset.py
    save_to_webdataset_main()

if __name__ == "__main__":
    run_all_scripts()