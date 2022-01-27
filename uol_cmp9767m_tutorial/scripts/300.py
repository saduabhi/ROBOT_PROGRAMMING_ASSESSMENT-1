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
from object_detection import ObjectDetection

def od():
    od = ObjectDetection()
    while True:
        # Detect objects on frame
        (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))
    
        # Initialize count
        count = 0
        center_points_prev_frame = []
        tracking_objects = {}
        track_id = 0

    # Only at the beginning we compare previous and current frame
        if count <= 2:
            for pt in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    continue
            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)
            else:
                tracking_objects_copy = tracking_objects.copy()
                center_points_cur_frame_copy = center_points_cur_frame.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue
            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)
        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1


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
            
            
        cv2.imshow("maskClose", self.maskClose)
        cv2.imshow("mask",mask)
        cv2.imshow("cam",img)
        print(len(conts))
        if cv2.waitKey(10) &0xFF ==ord('q'):
                    cv2.cap.release()
                    cv2.destroyAllWindows()
                
if __name__ =='__main__':
    rospy.init_node('count_fruits')
    cf = count_fruits()
    rospy.spin()