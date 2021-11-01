# Secret Knock

Secret Knock is going to be a smart home assitant that allows you to touchlessly verify your identity without the use of any biometrics. It will accomplish this through the use of physical passwords - poses and hand gestures known only to a select group of users. 

## Milestone 1 Usage

The program has two modes - recognizing the presence of a person (meant to be used with a camera that is facing a door in order to detect when someone is near) and a hand gesture recognizer. 

It defaults to the hand gesture mode. In this mode, there are no pre-trained gestures. Instead, hold your hand up in the gesture that you want to train, then hit the key "t" for train. The program will record the shape of your hand. For the best recall possible, keep your hand in the same shape but move it around the screen and change its orientation in relation to it. Without changing the shape of your hand, press "t" again to exit training mode. From now on, if your hand is in a shape similar to this while in recognition mode, it should recall the gesture. This is verified with the text in the top left of the screen.

To enter whole body detection mode, press "b" for body. To switch back to hands mode, press "h" for hands.