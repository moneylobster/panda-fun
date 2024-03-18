"""
open a webserver that shows the current readout from the realsense cam
on the main route
"""
import time
import numpy as np
import json

from flask import Flask

import pyrealsense2 as rs
import cv2


FRAMERATE=30
RES=(640,480)
DEPTH=False
CAM_CONFIG="../435_high_accuracy_mode.json"

INTENDED_FRAMERATE=10
INTENDED_RES=(320,240)


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

# setup advanced config
# with open(CAM_CONFIG) as f:
#     json_text=f.read()
# device = pipeline_profile.get_device()
# advanced_mode = rs.rs400_advanced_mode(device)
# advanced_mode.load_json(json_text)

################################################################################
## SERVER SETUP

app=Flask(__name__)

@app.route("/")
def getimg():
    # Start cam stream
    pipeline.start(config)
    time.sleep(1)
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not color_frame or not depth_frame:
        return "Couldn't get image :("

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

    cv2.imwrite("static/image0.png", images)
    pipeline.stop()
    
    return "<img src=static/image0.png>"

if __name__=="__main__":
    app.run(host="0.0.0.0")
