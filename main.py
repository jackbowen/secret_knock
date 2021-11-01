import cv2
import secretknock_hands as sk_hands
import secretknock_body as sk_body
import numpy as np

HANDS = 1
hand_gestures = []
temp_gesture = []
num_gestures = 0
BODY = 2
TEXT_ORIGIN = (5, 25)
TEXT_SCALE = .5
TEXT_COLOR = (150, 0, 0)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0) #opens the webcam
    
    hands = sk_hands.handDetector()
    body = sk_body.bodyDetector()
    
    detect_mode = HANDS
    training_mode = False # This will be used to do live training for gestures

    while True:
        success, img = cap.read()
        img_rgb = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)

        if detect_mode == HANDS:
            img_rgb, hand_features = hands.findHands(img_rgb, training_mode)
            
            if training_mode:
                cv2.putText(img_rgb,
                            f"Training mode: gesture {num_gestures + 1}", 
                            TEXT_ORIGIN, 
                            cv2.FONT_HERSHEY_COMPLEX, 
                            TEXT_SCALE, 
                            TEXT_COLOR)
                #hand_gestures[num_gestures].append(hand_features)\
                temp_gesture.append(hand_features)
            else:
                recognized_gesture = -1
                threshold = .22
                gesture_text = "No gesture found"
                if len(hand_features) > 0:
                    min_dist = 1
                    for i in range(len(hand_gestures)):
                        gesture = hand_gestures[i]

                        #compute distance 
                        np_features = np.array(hand_features)
                        dist = np.linalg.norm(np_features - gesture)

                        if dist < threshold and dist < min_dist:
                            recognized_gesture = i
                            min_dist = dist
                            gesture_text = f"Found gesture {i+1}"
                        print(f"Gesture {i} dist: {dist}")

                        i += 1
                        #recognize if less than threshold
                #gesture_text = f"Found gesture {i + 1}" if i >= 0 else "No gesture found"
                cv2.putText(img_rgb, 
                            f"Recognition mode // {gesture_text}", 
                            TEXT_ORIGIN, 
                            cv2.FONT_HERSHEY_COMPLEX, 
                            TEXT_SCALE, 
                            TEXT_COLOR)


        elif detect_mode == BODY:
            img_rgb, body_present = body.findBody(img_rgb, training_mode)

            display_text = "Somebody is by the door" if body_present else "Nobody detected"
            cv2.putText(img_rgb, 
                        display_text, 
                        TEXT_ORIGIN, 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        TEXT_SCALE, 
                        TEXT_COLOR)

        # show the imaage with any annotations
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        #cv2.imshow("Video", cv2.flip(img,1))
        cv2.imshow("Video", img)
  
        #user interaction
        key =  cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break

        elif key == ord('h'):
            detect_mode = HANDS
        
        elif key == ord('b'):
            detect_mode = BODY

        elif key == ord('t'):            
            training_mode = not training_mode

            # We've just started training mode, make sure there's a place for
            # us to write our extracted features
            if training_mode:
                temp_gesture = []
                #hand_gestures.append([])

            # We've just ended training mode, increment our gesture counter
            # so we don't overwrite anything whenever we train the next
            # gesture
            if not training_mode:
                hand_gestures.append(np.array(temp_gesture).mean(axis=0))
                #for feature_set in temp_gesture:
                #    feature_avg[0]
                num_gestures += 1
            

