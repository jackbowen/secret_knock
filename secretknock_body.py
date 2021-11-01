import cv2
import mediapipe as mp
import time
import numpy as np

class bodyDetector():
    def __init__(self, mode=False, smooth=False, detectCon=0.5, trackingCon=0.4):
        self.mode = mode
        self.smooth = smooth
        self.mpPose = mp.solutions.pose
        self.drawStyles = mp.solutions.drawing_styles
        self.detectCon = detectCon
        self.trackCon = trackingCon

        self.mpPose = mp.solutions.pose
        self.body = self.mpPose.Pose(static_image_mode=self.mode, 
                                     smooth_landmarks=self.smooth, 
                                     min_detection_confidence=self.detectCon, 
                                     min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findBody(self, img, is_training):
        results = self.body.process(img)
        #display_text = "Nobody present"
        body_present = False

        if results.pose_landmarks:
            #display_text = "Someone is by the door"
            body_present = True
        
            self.mpDraw.draw_landmarks(img, 
                                       results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=self.drawStyles.get_default_pose_landmarks_style())

        # write whether or not there is a body present
        return img, body_present