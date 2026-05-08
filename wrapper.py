import tempfile
import numpy as np
import os
from utils import find_files, get_file_paths, load_obj_vertices_faces
from pathlib import Path

def aionhmr_process_videos(video_file: str, output_path: str) -> None:
    """
    Process a video using AionHMR-b and save the results.

    Parameters:
        video_file: Path to the video file.
        output_path: Output file path for the processed results (excluding extension).
    """

    # Check if output folder exists, if not create it
    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)
    
    # Run the AionHMR-b demo from the command line interface
    with tempfile.TemporaryDirectory() as temp_dir:
        os.system(f'python aionhmr-b/demo.py --img_folder \'{video_file}\' --out_folder \'{temp_dir}\' --batch_size 1 --save_mesh --is_video')

        # Take the results from the individual frames and merge them into a single tensor
        # For each frame, AionHMR saves multiple .obj files (one for each person detected in the frame)
        # We will create a tensor of shape (person, frames, ...)
        # We will only use the first (best) detected person, thus we only need files ending in '0.obj'
        # as the files are named like 'frame_10_0.obj for the best detected person in frame 10.
        file_names = [filename for filename in os.listdir(temp_dir) if filename.endswith('0.obj')]
        file_names.sort()

        # Stack the .obj files into a tensor
        # We will use the vertex positions from the .obj files, which are stored in lines
        full_video_tensor = np.zeros((len(file_names), 6890, 3)) # AionHMR uses the SMPL model which has 6890 vertices, each with x,y,z coordinates
        for file_name in file_names:
            obj_path = os.path.join(temp_dir, file_name)
            vertices, _ = load_obj_vertices_faces(obj_path)
            full_video_tensor[file_names.index(file_name)] = vertices
        
        # Add a new dimension for the person, since we are only using one person, this dimension will have size 1
        full_video_tensor = np.expand_dims(full_video_tensor, axis=0) # Shape: (1, frames, 6890, 3), (person, frames, vertices, coordinates)

        # Save the full video tensor as a .npy file
        np.save(f"{output_path}.npy", full_video_tensor)
        print(f'Saved processed video tensor to {output_path}')

if __name__ == "__main__":
    # Get all filepaths
    filepaths_df = find_files(r'../webdav/Measuring with AI (Projectfolder)')

    # Process all videos
    for index in range(len(filepaths_df)):
        for camera in range(2):
            print(f"Processing video {index*2 + camera + 1} (index {index}, camera {camera}) out of {len(filepaths_df)*2}")

            try:
                _, _, _, video_file = get_file_paths(filepaths_df, rowsel=index, camsel=camera)
                aionhmr_process_videos(video_file, f'../data/aionhmr/raw_output/{Path(video_file).stem}')
            except Exception as e:
                print(f"Error processing video {index*2 + camera + 1} (index {index}, camera {camera}): {e}")
                with open("log.txt", "a") as log_file:
                    log_file.write(f"Error processing video {index*2 + camera + 1} (index {index}, camera {camera}): {e}\n")
