import pyrealsense2 as rs
import numpy as np
import cv2
h = 720
w = 1280
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 6)
#config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

# Start streaming
pipeline.start(config)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 6.0, (w,h))

frame_list = []

import os, sys

# Path to be created
if not os.path.exists('./jpeg' ):
    os.mkdir('./jpeg' );
if not os.path.exists('./png' ):
    os.mkdir('./png' );
        
try:
  for i in range(0, 6*60):
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth:
        continue
    
    depth_data = depth.as_frame().get_data()
    np_image = np.asanyarray(depth_data)
    #frame_list.append(np_image)
    img = np.zeros([np_image.shape[0],np_image.shape[1],3], dtype=np.uint8)
    x = np.array([np_image, np_image>>8], dtype=np.uint8 )
    img[:,:,0] = x[0]   
    img[:,:,1] = x[1]
    #img[:,:,2] = x[1]    
    y = np.array(img[:,:,1], dtype=np.uint16 ) << 8 
    np_image_r = y + np.array(img[:,:,0], dtype=np.uint16 )
    #img[:,:,2] = x[1]
    #img[:,:,1] = numpy.ones([5,5])*128/255.0
    #img[:,:,2] = numpy.ones([5,5])*192/255.0
    depth_colormap_r = cv2.applyColorMap(cv2.convertScaleAbs(np_image_r, alpha=0.3), cv2.COLORMAP_JET)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np_image, alpha=0.3), cv2.COLORMAP_JET)
    cv2.imwrite('jpeg/color_img_{0}.jpg'.format(i), img)
    cv2.imwrite('png/color_img_{0}.png'.format(i), img)
    out.write(img)
    images = np.hstack((depth_colormap, depth_colormap_r))
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)
    #cv2.imshow("image", img);
    #cv2.waitKey();
    for f in frames:
      print(f.profile)
finally:
    pipeline.stop()
    out.release()
    
    

