import cv2
import mediapipe as mp
import time
import numpy as np


cap = cv2.VideoCapture(0) #opens the webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=.5,
                      min_tracking_confidence=.3)
mpDraw = mp.solutions.drawing_utils

while True:
  success, img = cap.read()
  #img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #img_blur = cv2.GaussianBlur(img_gray,(3,3),0) #blur the image to reduce some noise
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = hands.process(img_rgb)

  # if we detect any hands, check further to see if we recognize any gestures
  if results.multi_hand_landmarks:

    # for each hand detected, check its landmarks
    for handLms in results.multi_hand_landmarks:

      landmarks = []
      # for each of the 21 landmarks in a hand, do something
      for id, lm in enumerate(handLms.landmark):
        #print(id,lm)
        landmarks.append([lm.x, lm.y, lm.z])
        pass

      print(landmarks[0])
      mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
      
  cv2.imshow("Video", cv2.flip(img,1))
  
  if cv2.waitKey(1) & 0xFF ==ord('q'):
    break
