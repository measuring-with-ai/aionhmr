# Read all files from demo_out_new that contain all in their filename
import os
import cv2

filenames = [f for f in os.listdir('demo_out_new_2') if 'all' in f]

# Create a video using these images
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('demo_out_new_2/demo_video.mp4', fourcc, 30.0, (1280, 720))
for filename in sorted(filenames):
    img = cv2.imread(os.path.join('demo_out_new_2', filename))
    out.write(img)
out.release()