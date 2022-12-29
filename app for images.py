import os, cv2, mediapipe as mp, numpy as np, pandas as pd, pickle, matplotlib.pyplot as plt
import random

mp_holistic = mp.solutions.holistic

# Load the model and the label encoder from the pickle file
model, label_encoder = pickle.load(open('model.pkl','rb'))

# Create the holistic object for pose detection
holistic = mp_holistic.Holistic(static_image_mode=True, model_complexity=2)

# Set the path to the directory containing the asana folders
path = "DATASET/TEST"

# Iterate through the asana folders
for asana in os.listdir(path):
    
    # Get the list of images in the asana folder
    images = os.listdir(os.path.join(path,asana))
    
    # Shuffle the list of images
    random.shuffle(images)
    
    # Take the first 3 images from the shuffled list
    images = images[:3]
    
    # Iterate through the images in the asana folder
    for image in images:
        
        # Set the path to the image
        image_path = os.path.join(path,asana,image)
        
        # Load the image
        img = cv2.imread(image_path)
        
        # Convert the image to RGB
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        # Detect the pose landmarks in the image
        results = holistic.process(imgRGB)
        
        # If pose landmarks were detected
        if results.pose_landmarks:
            
            # Create a data point from the pose landmarks
            row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())
            X = pd.DataFrame([row])
            
            # Predict the body language class of the person in the image
            body_language_class = label_encoder.inverse_transform(model.predict(X))[0]
            body_language_prob = model.predict_proba(X)[0]
            
            # If the probability of the prediction is high enough, display the predicted body language class on the image with a "Success" message
            if round(body_language_prob[np.argmax(body_language_prob)],2)*100 >= 70:
                cv2.putText(img, str("Success: "+ body_language_class), (0, 100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                cv2.putText(img, '{} {}'.format(body_language_class,int(round(body_language_prob[np.argmax(body_language_prob)],2)*100)), (100, 300),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            else:
                cv2.putText(img, str("Unknown Pose"), (0, 100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                cv2.putText(img, '{} {}'.format(body_language_class,int(round(body_language_prob[np.argmax(body_language_prob)],2)*100)), (100, 300),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            
            plt.imshow(img)
            plt.show()
            
#             if not os.path.exists("output_folder"):
#                 os.makedirs("output_folder")
            
#             cv2.imwrite(os.path.join("output_folder", image), img)