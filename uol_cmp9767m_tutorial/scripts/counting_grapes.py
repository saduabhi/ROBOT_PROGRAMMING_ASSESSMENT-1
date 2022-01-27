import cv2 # opencv
import numpy as np
from numpy import mean, array
# ROS Messages
from sensor_msgs.msg import Image
# ROS Libraries
from cv_bridge import CvBridge
import rospy
from cv2 import destroyAllWindows, startWindowThread
from cv2 import namedWindow, cvtColor, imshow, inRange
from cv2 import blur, Canny, resize, INTER_CUBIC
from cv2 import COLOR_BGR2GRAY, waitKey, COLOR_BGR2HSV


class count_fruits:
# Define
    def __init__(self): 
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/thorvald_001/kinect2_right_camera/hd/image_color_rect",Image, self.process_image)
        self.maximum = 0

    def process_image(self, camera):
        #cam= cv2.VideoCapture(0)
        self.kernelClose=np.ones((15,15))
        img = self.bridge.imgmsg_to_cv2(camera, "bgr8")
        img = resize(img, None, fx=0.5, fy=0.5, interpolation = INTER_CUBIC)
        #img=cv2.resize(img,(340,220))
        # Bounds are used for choosing the colour i.e Purple color in this case.
        lowerBound=np.array([100,18,46])
        upperBound=np.array([107,255,255])
        # Choosing the specific Font style
        font = cv2.FONT_HERSHEY_SIMPLEX

        #convert (Blue,Green,Red) BGR to (Hue,Saturation,value)HSV
        imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #cv2 Image show
        # cv2.imshow("image_mask", image_mask)
        cv2.imshow('HSV',imgHSV)
        # create the Mask
        mask=cv2.inRange(imgHSV,lowerBound,upperBound)
        #morphology
        self.maskClose=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,self.kernelClose)

        maskFinal=self.maskClose
        #find contours
        #dst=cv2.findContours(srcImageCopy, contourRetrievalMode, contourApproximationMethod)
        _,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #Draw contours
        cv2.drawContours(img,conts,-1,(255,255,255),1)
        for i in range(len(conts)):
            area = cv2.contourArea(conts[i])
            if area >10:
                x,y,w,h=cv2.boundingRect(conts[i])
                #cv2.rectangle(src,startpoint,endpoint,color,thickness)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
                #cv2.putText(src,text,position,font,fontSize,color,thickness)
                cv2.putText(img, str(i+1),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0))
            
         # cv2.imshow("image_mask", image_mask)   
        cv2.imshow("maskClose", self.maskClose)
        cv2.imshow("mask",mask)
        cv2.imshow("cam",img)
        print("Number_of grapes are")
        if len(conts) > self.maximum:
            self.maximum =len(conts)
        print(self.maximum)
        #waitkey commands when to stop.
        if cv2.waitKey(10) &0xFF ==ord('q'):
                    cv2.cap.release()
                    cv2.destroyAllWindows()
                
if __name__ =='__main__':
    rospy.init_node('count_fruits')
    cf = count_fruits()
    rospy.spin()