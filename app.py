import cv2,time
import math
import mediapipe as mp
import pyautogui
import numpy as np
import pickle
import pandas as pd

model=pickle.load(open('model.pkl','rb'))

# Initializing mediapipe pose class.
mp_holistic = mp.solutions.holistic

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    # Check if any landmarks are detected.
    if results.pose_landmarks :
        
        # Draw Pose landmarks on the output image.        
        # 1. Draw face landmarks
        '''mp_drawing.draw_landmarks(output_image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(output_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(output_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )'''
        
    # Return the output image and the found landmarks.
    return output_image, results

# Pose Estimation

# Setup Holistic Pose function for video.
pose_video = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

screen_width, screen_height = pyautogui.size()

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

fps = []

# Initialize a variable to store the time of the previous frame.
time1 = 0

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly.
    if not ok:

        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (screen_width,screen_height))

    # Perform Pose landmark detection.
    frame, results = detectPose(frame,pose_video ,display=False)

    # Check if the landmarks are detected.
    if results.pose_landmarks:
            row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            #print(body_language_class,round(body_language_prob[np.argmax(body_language_prob)],2)*100)
            if round(body_language_prob[np.argmax(body_language_prob)],2)*100 >= 70 :
                cv2.putText(frame, str("Success "+str(body_language_class,int(round(body_language_prob[np.argmax(body_language_prob)],2)*100))), (300, 100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                # 2. Right hand
                '''mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                                         )

                # 3. Left Hand
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                                         )'''

                # 4. Pose Detections
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                                         )
            else:
                cv2.putText(frame, str("Unknown Pose"), (300, 100),cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 3)
                cv2.putText(frame, '{} {}'.format(body_language_class,int(round(body_language_prob[np.argmax(body_language_prob)],2)*100)), (1200, 100),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                # 2. Right hand
                '''mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                         )

                # 3. Left Hand
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                         )'''

                # 4. Pose Detections
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                         )

    # Set the time for this frame to the current time.
    time2 = time.time()

    # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
    if (time2 - time1) > 0:

        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)

        fps.append(int(frames_per_second))

        # Write the calculated number of frames per second on the frame. 
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (0, 100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        #Write the predicted asana label on the frame.
        #cv2.putText(frame, str(body_language_class), (300, 100),cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2

    # Display the frame.
    cv2.imshow('Pose Classification', frame)

    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF

    if(k==27):
        break

camera_video.release()
cv2.destroyAllWindows()