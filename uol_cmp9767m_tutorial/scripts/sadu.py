# from builtins import breakpoint
from termios import VSTART
# import cv2
import cv2
import time
# import imutils
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy

class Count:
    firstFrame = None
    area = 10
    def __init__(self): 
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/thorvald_001/kinect2_right_camera/hd/image_color_rect",Image, self.process_image)

    def process_image(self, data):
        while True:
            grayImg = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
            gaussianImg = cv2.GaussianBlur(grayImg, (21,21), 0)
            if self.firstFrame is None:
                self.firstFrame = gaussianImg
                continue
            imgDiff = cv2.absdiff(self.firstFrame, gaussianImg)
            threshImg = cv2.threshold(grayImg,200,255,cv2.THRESH_BINARY) [1]
            threshImg = cv2.dilate(threshImg, None, iterations=2)
            cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
                if cv2.contourArea(c) > self.area:
                    continue
            (x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(src,startpoint,endpoint,color,thickness)
            cv2.rectangle(data,(x,y),(x+w, y+h),(0,250,0),2)
            text = "Moving Object detected"
            print(text)

            cv2.putText(data,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

if __name__ =='__main__':
    rospy.init_node('count_fruits')
    cf = Count()
    rospy.spin()