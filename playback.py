import numpy as np
import cv2

cap = cv2.VideoCapture('output.avi')
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    #depth_data = depth.as_frame().get_data()
    #np_image = np.asanyarray(depth_data)
    #frame_list.append(np_image)
    #img = np.zeros([np_image.shape[0],np_image.shape[1],3], dtype=np.uint8)
    #x = np.array([np_image, np_image>>8], dtype=np.uint8 )
    #img[:,:,0] = x[0]   
    #img[:,:,1] = x[1]
    y = np.array(frame[:,:,1], dtype=np.uint16 )<<8 
    np_image = y + np.array(frame[:,:,0], dtype=np.uint16 )
    #img[:,:,2] = x[1]
    #img[:,:,1] = numpy.ones([5,5])*128/255.0
    #img[:,:,2] = numpy.ones([5,5])*192/255.0
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np_image, alpha=0.3), cv2.COLORMAP_JET)
    #cv2.imwrite('color_img_{0}.jpg'.format(i), img)
    #out.write(img)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', depth_colormap)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
