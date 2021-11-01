import cv2
import mediapipe as mp
import time
import numpy as np

class handDetector():
    def __init__(self, mode=Fase, maxHands=2, detectCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectCon
        self.trackCon = trackingCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    #takes an rgb image
    def findHands(self, img):
        results = self.hands.process(img)

        # if we detect any hands, check further to see if we recognize any gestures
        if results.multi_hand_landmarks:

            # for each hand detected, check its landmarks
            for handLms in results.multi_hand_landmarks:

                landmarks = []
                # for each of the 21 landmarks in a hand, do something
                for id, lm in enumerate(handLms.landmark):
                    #print(id,lm)
                    landmarks.append([lm.x, lm.y, lm.z])

            print(landmarks[0])
            self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
