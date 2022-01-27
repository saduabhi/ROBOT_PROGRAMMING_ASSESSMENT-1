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
import math

# Initialize count
#count = 0
# centre_pt_prev_frame = []
# tracking_obj = {}
# track_id = 0

# while True:

#     centre_pt_cur_frame = []



class count_fruits:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/thorvald_001/kinect2_right_camera/hd/image_color_rect", Image,
                                          self.process_image)

    def process_image(self, camera):
        centre_pt_cur_frame = []
        centre_pt_prev_frame = []
        tracking_obj = []
        track_id = 0
        centre_pt_prev_frame = []

        # cam= cv2.VideoCapture(0)
        self.kernelClose = np.ones((15, 15))
        img = self.bridge.imgmsg_to_cv2(camera, "bgr8")
        img = resize(img, None, fx = 0.6, fy = 0.6, interpolation = INTER_CUBIC)
        # img=cv2.resize(img,(340,220))

        lowerBound = np.array([100, 18, 46])
        upperBound = np.array([107, 256, 256])
        font = cv2.FONT_HERSHEY_SIMPLEX

        # convert BGR to HSV
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV', imgHSV)
        # create the Mask
        mask = cv2.inRange(imgHSV, lowerBound, upperBound)
        # morphology
        self.maskClose = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernelClose)

        maskFinal = self.maskClose
        _, conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(img, conts, -1, (255, 255, 255), 1)
        for i in range(len(conts)):
            area = cv2.contourArea(conts[i])
            
            if area > 10:
                #x - co ordinates
                #y -co ordinates
                #w - width
                #h - height
                #(x,y) left most co-ordinates
                #(x + w, y + h) - bottom co-ordinates
                #(0, 0, 255), 2 - color and thickness of the image
                x, y, w, h = cv2.boundingRect(conts[i])
                # cx and cy are to find the centre of the image
                cx = ((x + x + w) / 2)
                cy = ((y + y + h) / 2)

                # This is used to store the all the previous points
                centre_pt_cur_frame.append((cx, cy))

                print(" The co-ordinates of the grapes:", x, y, w, h)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, str(i + 1), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
                cv2.circle(img, (cx,cy), 5, (0, 0, 255), -1)

       # only at the begining we compare previous and current frame
            if conts <= 20:
               for pt in centre_pt_cur_frame:
                   for pt2 in centre_pt_prev_frame:
                       distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
    
                       if distance<20:
                        tracking_obj[track_id] = pt
                        track_id=+1
    
            else:
               tracking_obj_cpy = tracking_obj.copy()
               centre_pt_cur_frame_cpy = centre_pt_cur_frame.copy()
    
               for object_id, pt2 in tracking_obj_cpy.items():
               
                   object_exists = False
    
                   for pt in centre_pt_cur_frame_cpy:
                        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                        #update id position
                        if distance < 20:
                            tracking_obj[object_id] = pt
                            object_exists = True
                            centre_pt_cur_frame.remove(pt)
                            continue
                        
                   #remove the Id lost
                        if not object_exists:
                          tracking_obj.pop(object_id)
                 # Add new ids found
               for pt in centre_pt_cur_frame:
               
                   tracking_obj[track_id] = pt
                   track_id = track_id + 1





        for object_id, pt in tracking_obj:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)
            cv2.putText(img, str(object_id), (pt[0], pt[1]-7), 0, 1, (0,0,255), -2)


 

        print("Tracking objects", tracking_obj)
        print(" Current left frame: ", centre_pt_cur_frame)




        # Make a copy of the points

        centre_pt_prev_frame = centre_pt_cur_frame


        cv2.imshow("MaskClose", self.maskClose)
        cv2.imshow("Mask", mask)
        cv2.imshow("Camera", img)
        print("Total number of grapes")
        print(len(conts))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
     rospy.init_node('count_fruits')
     cf = count_fruits()
     rospy.spin()