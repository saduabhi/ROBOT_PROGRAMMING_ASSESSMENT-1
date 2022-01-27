#from builtins import breakpoint
import cv2
from termios import VSTART
import time
#import imutils
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy


#Reading frame from camera

def cam():
    cam =cv2.Videocapture(0)
#time.sleep(5)

#firstFrame=None
#area =20
def img():
    while True:
        _,img = cam.read()
        text ="Normal"
        img = imutils.resize(img, width=20)
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gaussianImg = cv2.GaussianBlur(grayImg, (21,21), 0)
        if firstFrame is None:
            firstFrame = gaussianImg
            continue
        imgDiff = cv2.absdiff(firstFrame, gaussianImg)
        thresImg = cv2.threshold(grayImg,200,255,cv2.THRESH_BINARY) [1]
        thresImg = cv2.dilate(threshImg, None, iterations=2)

    #find contours
    #dst=cv2.findContours(srcImageCopy, contourRetrievalMode, contourApproximationMethod)
        cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            if cv2.contourArea(c) > area:
                 continue
            (x, y, w, h) = cv2.boundingRect(c)
            
            #cv2.rectangle(src,startpoint,endpoint,color,thickness)
            cv2.rectangle(img,(x,y),(x+w, y+h),(0,250,0),2)
            text = "Moving Object detected"
            print(text)

        #cv2.putText(src,text,position,font,fontSize,color,thickness)
        cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        cv2.imshow("Videostream",img)
        cv2.imshow("Thresh",threshImg)
        cv2.imshow("Imagedifference",imgDiff)
        key = cv2.waitKey(1)&0xFF
        if key == ord("q"):
            break

def vs():
    vs.release()
    cv2.destroyAllWindows()

