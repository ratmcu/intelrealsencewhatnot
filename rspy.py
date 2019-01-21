import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

# Start streaming
pipeline.start(config)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 6.0, (1280,720))

frame_list = []
 
try:
  for i in range(0, 6*60):
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    depth_data = depth.as_frame().get_data()
    np_image = np.asanyarray(depth_data)
    frame_list.append(np_image)
    img = np.zeros([np_image.shape[0],np_image.shape[1],3], dtype=np.uint8)
    x = np.array([np_image, np_image>>8], dtype=np.uint8 )
    img[:,:,0] = x[0]   
    img[:,:,1] = x[1]
    #img[:,:,1] = numpy.ones([5,5])*128/255.0
    #img[:,:,2] = numpy.ones([5,5])*192/255.0

    #cv2.imwrite('color_img_{0}.jpg'.format(i), img)
    out.write(img)
    #cv2.imshow("image", img);
    #cv2.waitKey();
    for f in frames:
      print(f.profile)
finally:
    pipeline.stop()
    out.release()
    
    

