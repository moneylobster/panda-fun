'''
operate arm and record as binarized np array along with camera

this version records the camera and the robot together
'''
# TODO fix panda logging part to be dict
import numpy as np
import cv2
import time
import sys
from datetime import datetime

import panda_py

import pyrealsense2 as rs

if len(sys.argv)==1:
  raise RuntimeError("Please specify recording length.")
LEN=int(sys.argv[1])

FRAMERATE=30
RES=(640,480)
DEPTH=False

INTENDED_FRAMERATE=10
INTENDED_RES=(96,96)

FRAME_STRIDE=FRAMERATE//INTENDED_FRAMERATE
PERIOD=1/INTENDED_FRAMERATE

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
    print("Can't detect color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, RES[0], RES[1], rs.format.z16, FRAMERATE)
config.enable_stream(rs.stream.color, RES[0], RES[1], rs.format.bgr8, FRAMERATE)

imagelog=[]

################################################################################
## ARM SETUP

# connect to robot
panda=panda_py.Panda("10.0.0.2")
panda.move_to_start()

log={"q":[],"O_T_EE":[]}

################################################################################
## START

input(f'Teach a trajectory for {LEN} seconds. Press enter to begin.')
# Start cam stream
pipeline.start(config)

panda.teaching_mode(True)
# panda.enable_logging(LEN * 1000)
t_start=time.time()
while time.time()-t_start<=LEN:
    t_lastit=time.time()
    # Ask for a coherent pair of frames: depth and color
    frames = pipeline.poll_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not color_frame or not depth_frame:
        continue

    # Convert images to numpy arrays
    if DEPTH:
      depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    if DEPTH:
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
    else:
      images=color_image
    
    # save img
    imagelog.append(images)
    # save log
    log.append(panda.q)

    # Pause for the rest of the period
    while time.time()-t_lastit<PERIOD:
        pass

panda.teaching_mode(False)
print("Recording has ended.")
################################################################################
## END

# log=panda.get_log()
print(log.keys())

print(f"log {len(log['q'])}")
print(f"images {len(imagelog)} {imagelog[0].shape}")

# process the images to fit resolution and framerate
imagelog=[cv2.resize(img, INTENDED_RES) for img in imagelog[::FRAME_STRIDE]]

# save the log
filename=f"data/{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
np.save(filename+"_act.npy",log)
np.save(filename+"_obs.npy", imagelog)
print(f"Saved log as {filename}_act and images as {filename}_obs")
