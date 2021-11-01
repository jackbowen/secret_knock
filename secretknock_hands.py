import cv2
import mediapipe as mp
import time
import numpy as np
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectCon=0.5, trackCon=0.4, displayFlag=1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon
        self.displayFlag = displayFlag
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    #takes an rgb image
    def findHands(self, img, is_training):
        features = []

        results = self.hands.process(img)

        # if we detect any hands, check further to see if we recognize any gestures
        if results.multi_hand_landmarks:

            hands_landmarks = []
            # for each hand detected, check its landmarks
            for handLms in results.multi_hand_landmarks:

                # build a list of the 21 landmarks in a hand
                hand_landmarks = []
                for id, lm in enumerate(handLms.landmark):
                    #print(id)
                    hand_landmarks.append([lm.x, lm.y, lm.z])

                # add that list to all the hands we've detected
                hands_landmarks.append(hand_landmarks)

                features = self.extract_scaled_landmarks(hand_landmarks)

                # draw the landmarks on top of our image
                if (self.displayFlag == 1):
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)    

        return img, features
            

    
    def extract_scaled_landmarks(self, landmarks):
        # The landmarks are already normalized in relation 
        # The landmarks returned by mediapipe are normalized with 0 being the 
        # top left corner of the image and 1 being the bottom right. 0 in the z
        # direction is at the wrist and the scale is approximately the same in
        # that direction. The issue with this for feature extraction is that
        # any points we try to pull out would only match labeled images if the 
        # hand was located at the same point in the screen. Similarly, it would
        # need to be oriented along its 3 axes in the same way the labeled
        # image is. We want to extract a feature set that is agnostic of
        # position within the image or rotation. Thus, I'm just going to pull
        # out the distances from each of the landmarks in relation to an origin
        # on the hand, the bottom of the palm in this case. 

        origin = landmarks[0]
        
        lm_dists = []
        for lm in landmarks:
            lm_dist = math.sqrt((lm[0]-origin[0])**2 + 
                                (lm[1]-origin[1])**2 + 
                                (lm[2]-origin[2])**2)
            lm_dists.append(lm_dist)
        return lm_dists
