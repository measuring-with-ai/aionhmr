import numpy as np
import argparse
import os
import cv2
from utils import extract_frames_from_avi, find_files, get_file_paths

def aionhmr_process_video(frames: np.ndarray, output_folder: str) -> None:
    """
    Process a video using AionHMR-b and save the results.

    Parameters:
        frames: numpy array of shape (N, H, W, 3), BGR uint8 — same format as cv2.imread().
        output_folder: Path to the folder where results will be saved.
    """
    # Save the individual frames as images to temporary directory
    temp_dir = 'temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    try:
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f'frame_{i:06d}.jpg')
            cv2.imwrite(frame_path, frame)

        # Check if output folder exists, if not create it
        os.makedirs(output_folder, exist_ok=True)
        
        # Run the AionHMR-b demo from the command line interface
        os.system(f'python aionhmr-b/demo.py --img_folder {temp_dir} --out_folder {output_folder} --batch_size 1 --full_frame --save_mesh')
    finally:
        # Clean up temporary frames
        for temp_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, temp_file))
        os.rmdir(temp_dir)

ROOT_DIRECTORY = r'../webdav/Measuring with AI (Projectfolder)'

filename_df = find_files(ROOT_DIRECTORY)

_, _, _, avi_video = get_file_paths(filename_df, 0, 0)

frames, _ = extract_frames_from_avi(avi_video)

aionhmr_process_video(frames, 'demo_out_new_2')
