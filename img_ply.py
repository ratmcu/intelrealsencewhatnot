import numpy as np
import cv2

cap = cv2.VideoCapture('output.avi')
mse_png_vs_jp = 0
mse_png_vs_xvid = 0

for i in range(360):
    frame_jp = cv2.imread('jpeg/color_img_{0}.jpg'.format(i))
    frame_pn = cv2.imread('png/color_img_{0}.png'.format(i))
    ret, frame_xvid = cap.read()
    #depth_data = depth.as_frame().get_data()
    #np_image = np.asanyarray(depth_data)
    #frame_list.append(np_image)
    #img = np.zeros([np_image.shape[0],np_image.shape[1],3], dtype=np.uint8)
    #x = np.array([np_image, np_image>>8], dtype=np.uint8 )
    #img[:,:,0] = x[0]   
    #img[:,:,1] = x[1]
    y = np.array(frame_jp[:,:,1], dtype=np.uint16 )<<8 
    np_image_jp = y + np.array(frame_jp[:,:,0], dtype=np.uint16 )
    y = np.array(frame_pn[:,:,1], dtype=np.uint16 )<<8 
    np_image_pn = y + np.array(frame_pn[:,:,0], dtype=np.uint16 )
    y = np.array(frame_xvid[:,:,1], dtype=np.uint16 )<<8 
    np_image_xvid = y + np.array(frame_xvid[:,:,0], dtype=np.uint16 )
    #img[:,:,2] = x[1]
    #img[:,:,1] = numpy.ones([5,5])*128/255.0
    #img[:,:,2] = numpy.ones([5,5])*192/255.0
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np_image_jp, alpha=0.3), cv2.COLORMAP_JET)
    depth_colormap_r = cv2.applyColorMap(cv2.convertScaleAbs(np_image_pn, alpha=0.3), cv2.COLORMAP_JET)
    depth_colormap_s = cv2.applyColorMap(cv2.convertScaleAbs(np_image_xvid, alpha=0.3), cv2.COLORMAP_JET)
    #cv2.imwrite('color_img_{0}.jpg'.format(i), img)
    #out.write(img)
    images = np.hstack((depth_colormap, depth_colormap_r, depth_colormap_s))
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    mse_png_vs_jp += ((np_image_pn - np_image_jp)**2).mean()
    mse_png_vs_xvid += ((np_image_pn - np_image_xvid)**2).mean()
    #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('RealSense', depth_colormap)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print('error in jpeg {0}'.format(mse_png_vs_jp/(360)))
print('error in xvid {0}'.format(mse_png_vs_xvid/(360)))
