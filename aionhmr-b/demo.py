from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from src.configs import CACHE_DIR_AIONHMR
from src.models import AionHMRb, download_models, load_aionhmr, DEFAULT_CHECKPOINT
from src.utils import recursive_to
from src.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from src.utils.renderer import Renderer, cam_crop_to_full

# As we run the AionHMR-b demo code using the command line, it cannot do relative imports from 
# the utils.py file, so we need to copy the extract_frames_from_avi function here as well. In the 
# future, it would be better to refactor the code so that we can avoid this code duplication, but 
# for now this quick fix will do

# from ..utils import extract_frames_from_avi
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

import time

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='AionHMR-b demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--is_video', dest='is_video', action='store_true', default=False, help='If set, input is an AVI video file instead of image folder')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--per_person_render', dest='per_person_render', action='store_true', default=False, help='If set, save per-person renders')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_AIONHMR)
    model, model_cfg = load_aionhmr(args.checkpoint)

    # Setup AionHMR-b model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from src.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == 'vitdet':
        from detectron2.config import LazyConfig
        import src
        cfg_path = Path(src.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # If input is video, we treat the video file path as the only "image path", and then we will extract frames from the video later. If input is image folder, we get all image paths as usual. 
    # We use this approach because it allows us to reuse the same code for both videos and images.
    if args.is_video:
        img_paths = [args.img_folder]
    else:
        # Get all demo images that end with .jpg or .png
        img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    
    # For videos, we use an optimized way of extracting frames using a single pass through the video, instead of reading the video multiple times for each frame. 
    if args.is_video:
        new_img_paths = []
        img_cv2s = []
        img_path = img_paths[0] # Get the single video file path
        frames, _ = extract_frames_from_avi(str(img_path))
        for i, frame in enumerate(frames):
            new_img_path = f'{img_path}_frame_{i:04d}.png'
            new_img_paths.append(new_img_path)
            img_cv2s.append(frame)
        
        # We require the img_paths because it is used to get the filename for saving results, but we can skip reading the images again since we already have them in img_cv2s
        img_paths = new_img_paths
    else:
        img_cv2s = [cv2.imread(str(img_path)) for img_path in img_paths]

    # Iterate over all images in folder
    for img_path, img_cv2 in zip(img_paths, img_cv2s):

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Run AionHMR-b on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            start = time.time()
            with torch.no_grad():
                out = model(batch)
            end = time.time()
            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                if args.side_view or args.top_view or args.per_person_render or args.full_frame:
                    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([final_img, side_img], axis=1)

                if args.top_view:
                    top_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            top_view=True)
                    final_img = np.concatenate([final_img, top_img], axis=1)

                if args.per_person_render:
                    cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])

        end = time.time()
        print(end - start)

if __name__ == '__main__':
    main()
