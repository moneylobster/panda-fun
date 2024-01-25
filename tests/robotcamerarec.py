import numpy as np
import cv2
import time
import sys
from datetime import datetime

import panda_py
from panda_py import controllers
import spatialmath.base as smb

import pyrealsense2 as rs

if len(sys.argv)==1:
  raise RuntimeError("Please specify recording length.")
LEN=sys.argv[1]

################################################################################
## CAMERA SETUP

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

imagelog=[]
    
################################################################################
## ARM SETUP

# connect to robot
panda=panda_py.Panda("10.0.0.2")
panda.move_to_start()

################################################################################
## START

input(f'Teach a trajectory for {LEN} seconds. Press enter to begin.')
# Start cam stream
pipeline.start(config)

panda.teaching_mode(True)
panda.enable_logging(LEN * 1000)
t_start=time.time()
while time.time()-t_start<=LEN:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape
    
    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))
    
    # save img
    imagelog.append(images)

panda.teaching_mode(False)
print("Recording has ended.")
################################################################################
## END

log=panda.get_log()
print(log.keys())
print(f"endeffs: {endeffs[0:2]} total size:{endeffs.shape}")
print(f"pos/rot for first: {poss[0]} {rots[0]}")

print(f"log {log.shape}")
print(f"images {len(imagelog)} {imagelog[0].shape}")

# # save the log
# filename=f"log_{datetime.now().isoformat()}.npy"
# np.save(filename,log)
# print(f"Saved log as {filename}")
