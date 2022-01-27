#BGR to HSV
#dst = cv2.cvtColor(src, cv2.COLOR_BGR"HSV)
##hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#minimum enclosing circle
#((x,y),radius) = cv2.minEnclosingCircle(contourArea)
##((x,y),radius) = cv2.minEnclosingCircle(c)

#moments to find center of the area
##M = cv2.moments(c)

##center = (int(M["m10"]/ M["m00"]), int(M["m01"]/M["m00"]))
# Cx = M10/M00 , Cy = M01/M00

#Drawing circles
#cv2.circle(src,(x,y), int(radius),colour,thickness)
##cv2.circle(frame, (int(x), int(y)), int(radius),(0,255,255),2)

#cv2.circle(frame,center,5,(0,0,255), -1)
##cv2.circle(frame, center,5,(0,0,255), -1)

#colour calibration

import cv2
import numpy as np
from numpy import mean, array
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
from cv2 import namedWindow, cvtColor, imshow, inRange
from cv2 import destroyAllWindows, startWindowThread
from cv2 import COLOR_BGR2GRAY, waitKey, COLOR_BGR2HSV
from cv2 import blur, Canny, resize, INTER_CUBIC

class count_fruits:

    def __init__(self): 
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/thorvald_001/kinect2_right_camera/hd/image_color_rect",Image, self.process_image)

    def process_image(self, camera):


        #cam= cv2.VideoCapture(0)
        self.kernelClose=np.ones((15,15))
        img = self.bridge.imgmsg_to_cv2(camera, "bgr8")
        img = resize(img, None, fx=0.6, fy=0.6, interpolation = INTER_CUBIC)
        #img=cv2.resize(img,(340,220))

        lowerBound=np.array([101,18,46])
        upperBound=np.array([109,256,256])
        font = cv2.FONT_HERSHEY_SIMPLEX

        #convert BGR to HSV
        imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV',imgHSV)
        # create the Mask
        mask=cv2.inRange(imgHSV,lowerBound,upperBound)
        #morphology
        self.maskClose=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,self.kernelClose)

        maskFinal=self.maskClose
        _,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        cv2.drawContours(img,conts,-1,(255,255,255),1)
        for i in range(len(conts)):
            area = cv2.contourArea(conts[i])
            if area >10:
                x,y,w,h=cv2.boundingRect(conts[i])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
                cv2.putText(img, str(i+1),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0))
        #obtained the code from
        for c in conts:

            M = cv2.moments(c)
            if M["m00"] == 0:
                print ('No object detected.')
                return
                # calculate the y,x centroid
            image_coords = (M["m01"] / M["m00"], M["m10"] / M["m00"])
            # "map" from color to depth image
            depth_coords = (image_depth.shape[0]/2 + (image_coords[0] - image_color.shape[0]/2)*self.color2depth_aspect,
                image_depth.shape[1]/2 + (image_coords[1] - image_color.shape[1]/2)*self.color2depth_aspect)
            # get the depth reading at the centroid location
            depth_value = image_depth[int(depth_coords[0]), int(depth_coords[1])] # you might need to do some boundary checking first!            
            
            cv2.imshow("maskClose", self.maskClose)
            cv2.imshow("mask",mask)
            cv2.imshow("cam",img)
            print(len(conts))
            if cv2.waitKey(10) &0xFF ==ord('q'):
                cv2.cap.release()
                cv2.destroyAllWindows()
                    
    def main(args):
        '''Initializes and cleanup ros node'''
        rospy.init_node('count_fruits')
        cf = count_fruits()
        rospy.init_node('image_projection', anonymous=True)
        ic = image_projection()
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print ("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)