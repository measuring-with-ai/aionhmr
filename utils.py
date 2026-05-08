import cv2
import numpy as np
import os
import pandas as pd
import re

def extract_frames_from_avi(file_path: str) -> tuple[np.ndarray, float]:
    '''
    Extracts frames from an AVI video file and returns them as a NumPy array along with the frames per second (FPS).
    
    Parameters:
        file_path (str): Path to the AVI video file.
        
    Returns:
        tuple[np.ndarray, float]: A tuple containing:
            frames (np.ndarray): A NumPy array of shape (num_frames, height, width, channels) containing the video frames.
            fps (float): The frames per second of the video.
    '''
    # Open the video file
    cap = cv2.VideoCapture(file_path)
    
    # Handle the case where the video file cannot be opened
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {file_path}")
    
    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Extract frames
    frames = []
    while True:
        # Read next frame
        ret, frame = cap.read()

        # If ret is False, it means we've reached the end of the video or there was an error reading a frame
        if not ret:
            break

        # Append the frame to the list of frames
        frames.append(frame)
    
    # Release the video capture object
    cap.release()

    return np.array(frames), fps

def find_files(root_directory: str) -> pd.DataFrame:
    '''
    Scans the root directory for .c3d files and attempts to find corresponding 
    .xcp, .pkl, and video files in the Vicon subdirectory. Returns a DataFrame 
    summarizing the findings.
    
    
    Parameters:
        root_directory (str): The root directory to scan for .c3d files.

    Returns:
        pd.DataFrame: A DataFrame with columns for subdirectory, Vicon directory,
                      C3D file, XCP file, MOSH file, and video files.
    '''

    # Check for cached version
    if os.path.exists('cache/file_paths.csv'):
        os.makedirs('cache', exist_ok=True)
        results_df = pd.read_csv('cache/file_paths.csv')
        return results_df

    # The Vicon directory is assumed to be at a fixed relative path from the root directory
    vicon_directory = os.path.join(root_directory, 'intern/CMAS VICON-2021')

    header = ['Subdirectory', 'Vicondirectory', 'C3D File', 'xcp file', 'MOSH', 'video_cam1', 'video_cam2', 'WHAM_cam1', 'WHAM_cam2']
    results = []
    
    # --- Step 1: Pre-scan the Vicon Directory ---
    # Create a dictionary for fast lookup: {filename: folder_path}
    # This handles the unstructured nature of the Vicon directory.
    vicon_file_lookup = {}
    for v_subdir, _, v_files in os.walk(vicon_directory):
        for v_file in v_files:
            vicon_file_lookup[v_file] = v_subdir

    # --- Step 2: Main Processing Loop ---
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".c3d"):
                base_name = os.path.splitext(file)[0]
                
                # Patterns
                base_name_pattern = re.sub(r'\s+', '_', base_name)  # For .pkl (MOSH)
                c3d_name_pattern = re.sub(r'\s+', ' ', base_name)   # For WHAM folders
                
                derived_pkl = None
                derived_folders = []
                
                # 2a. Find MOSH file and WHAM folders in the current subdirectory
                for item in os.listdir(subdir):
                    item_path = os.path.join(subdir, item)
                    
                    if os.path.isfile(item_path) and item.endswith(".pkl"):
                        if re.match(f"{base_name_pattern}_stageii.pkl", item):
                            derived_pkl = item
                            
                    elif os.path.isdir(item_path):
                        # Find folders starting with the C3D name (WHAM outputs)
                        if item.startswith(c3d_name_pattern):
                            derived_folders.append(item)

                # 2b. Match WHAM folders to AVI files
                # We create pairs of (wham_folder, avi_filename) based on exact name matching
                matched_pairs = []
                for wham_folder in derived_folders:
                    expected_avi = f"{wham_folder}.avi"
                    
                    # Check if this specific AVI exists anywhere in the Vicon directory
                    if expected_avi in vicon_file_lookup:
                        matched_pairs.append((wham_folder, expected_avi))
                
                # Sort the pairs (usually by timestamp in filename) to ensure consistent Cam1/Cam2 assignment
                matched_pairs.sort()
                
                # 2c. Find the XCP file
                xcp_file = f"{base_name}.xcp"
                # Default vicon path to the root, update if we find the specific XCP file
                found_vicon_path = vicon_directory 
                
                if xcp_file in vicon_file_lookup:
                    found_vicon_path = vicon_file_lookup[xcp_file]
                else:
                    xcp_file = None # Will be dropped later if strictly required
                
                # 2d. Populate Result Row
                results.append({
                        'Subdirectory': subdir,
                        'Vicondirectory': found_vicon_path,
                        'C3D File': file,
                        'xcp file': xcp_file,
                        'MOSH': derived_pkl,
                        
                        # Unpack matched pairs. If a pair is missing, value is None.
                        'WHAM_cam1': matched_pairs[0][0] if len(matched_pairs) > 0 else None,
                        'video_cam1': matched_pairs[0][1] if len(matched_pairs) > 0 else None,
                        
                        'WHAM_cam2': matched_pairs[1][0] if len(matched_pairs) > 1 else None,
                        'video_cam2': matched_pairs[1][1] if len(matched_pairs) > 1 else None,
                    })
    
    # Covert results from list of dicts to DataFrame
    results_df = pd.DataFrame(results, columns=header)
    
    # Remove rows where crucial files (like XCP or videos) might be missing
    results_df.dropna(axis=0, inplace=True)

    # Cache the results for future runs
    os.makedirs('cache', exist_ok=True)
    results_df.to_csv('cache/file_paths.csv', index=False)

    return results_df

def get_file_paths(results_df: pd.DataFrame, rowsel: int, camsel: int) -> tuple[str, str, str, str]:
    '''
    Retrieves file paths for the Vicon XCP file, C3D marker file, MOSH results, BEV output, and AVI video based on the selected row and camera.

    Parameters:
        results_df (pd.DataFrame): The DataFrame containing the file paths and metadata.
        rowsel (int): The index of the row to select from the DataFrame.
        camsel (int): The index of the camera to select (0 for cam1, 1 for cam2).
    Returns:
        tuple[str, str, str, str, str]: A tuple containing the file paths for the XCP file, C3D file, MOSH file, and AVI video.
    '''

    # Get Vicon file paths (XCP), corresponding files for markers (C3D), and MOSH results for the selected row
    vicon_file_path = os.path.join(results_df.iloc[rowsel]['Vicondirectory'],results_df.iloc[rowsel]['xcp file'])
    markers_file_path = os.path.join(results_df.iloc[rowsel]['Subdirectory'],results_df.iloc[rowsel]['C3D File'])
    mosh_path = os.path.join(results_df.iloc[rowsel].Subdirectory, results_df.iloc[rowsel]['MOSH'])

    # Determine the camera and corresponding video file based on the selected camera
    if (camsel==0): 
        videocam = 'video_cam1'
    else: 
        videocam = 'video_cam2'

    # Get the AVI video file path based on the selected camera
    avi_file = os.path.join(results_df.iloc[rowsel]['Vicondirectory'],results_df.iloc[rowsel][videocam])

    return vicon_file_path, markers_file_path, mosh_path, avi_file

def load_obj_vertices_faces(path: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Loads vertices and faces from an OBJ file.

    Parameters:
        path (str): Path to the OBJ file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: An array of shape (num_vertices, 3) containing the vertex positions.
            - np.ndarray: An array of shape (num_faces, 3) containing the vertex indices for each face.
    '''
    vertices = []
    faces = []

    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue

            parts = line.split()
            tag = parts[0]

            # vertex position
            if tag == 'v':
                # v x y z
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])

            # face (supports v, v/t, v//n, v/t/n)
            elif tag == 'f':
                face = []
                for v in parts[1:]:
                    # e.g. "3/2/1" -> "3"
                    vid = v.split('/')[0]
                    face.append(int(vid) - 1)  # OBJ is 1-based, NumPy 0-based
                # if the face is not a triangle, you might want to triangulate here
                if len(face) == 3:
                    faces.append(face)
                # simple fan triangulation for n-gons:
                elif len(face) > 3:
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i+1]])

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces