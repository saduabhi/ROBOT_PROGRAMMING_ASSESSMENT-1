import numpy as np
import cv2
def resize(img):
        return cv2.resize(img,(512,512)) # arg1- input image, arg- output_width, output_height
def cap():
    cap=cv2.VideoCapture(vid_file_path)
    ret,frame=cap.read()

    ret,frame=cap.read()
    l_b=np.array([0,230,170])# lower hsv bound for red
    u_b=np.array([255,255,220])# upper hsv bound to red
def ret():

    while ret==True:
        ret,frame=cap.read()

        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,l_b,u_b)

        contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        max_contour = contours[0]
def contours():

    for contour in contours:


                if cv2.contourArea(contour)>cv2.contourArea(max_contour):
                    max_contour = contour
                    approx=cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True),True)
                    x,y,w,h=cv2.boundingRect(approx)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)

                M=cv2.moments(contour)
                cv2.imshow("frame",resize(frame))

                cv2.imshow("mask",mask)
                key=cv2.waitKey(1)
                if key==ord('q'):
                    break
cv2.waitKey(0)
cv2.destroyAllWindows()
