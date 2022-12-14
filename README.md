# ShapeWell-Machine-Learning
A Virtual Yoga Asana Assistant using Machine Learning.

This application using Machine Learning and some python libraries to detect some Yoga asanas which were pre trained to the model.

It contains 3 stages for the working of the application.

They are 
1) Data Collection
2) Data Training 
3) App

## 1) Data Collection
It is a process of collection of landmarks of joints in a human body. To perform this process we use two modules called OpenCV, Mediapipe.

Image dataset of Yoga Asana is taken from here - https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset

### a) OpenCV - It is used to perform operations on images like reading, writing etc. For more information please refer to this - https://opencv.org/

![image](https://user-images.githubusercontent.com/57896227/206229705-4bd49e14-af95-4cab-b230-89347905d041.png)
                                                    
                                                     Fig:- OpenCV Face Detection (Sample Example)

### b) Mediapipe - It is used to perform landmark detection of keypoints in a human body. For more information please refer to this - https://mediapipe.dev/

![image](https://user-images.githubusercontent.com/57896227/206228479-c8fd39f8-58a1-43de-8539-9c1f6e880caf.png)

                                                     Fig:- Mediapipe Holistic Pipeline Sample
                                                     
After performing all these steps all the landmarks of the body keypoints are stored in a.csv file which is used for further processing.

## 2) Data Training
It is a process of reading, extracting the values from .csv file.These extracted values are used for training our ML model. Here we use Random Forest Algorithm as best optimal solution as it fits our problem. The trained model will be saved by using pickle module which will be used for predicting the asanas in the real time.

## 3) App
Final stage of the process where we use the pre-trained ML model for predicting Yoga asanas in the real time using Webcam feed of the device.It shows the landmarks of the keypoints in red colour if the machine is not able to recognize the asana (Incoorect way of doing) and turns into green colour if the machine recognizes the asana (Correct way of doing).

![Screenshot (25)](https://user-images.githubusercontent.com/57896227/210140917-eb185108-9fe2-41c6-8794-709a4909834f.png)

                                               Fig:- Machine detects the Asana (Correct way of doing)
