import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime
import os
import pyttsx3

engine = pyttsx3.init()
text = 'Bad posture, Arnav'

mp_pose = mp.solutions.pose
pose_estimation_params = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
last_logged_time = time.time()
log_interval = 5*60
last_alert_time = time.time()
alert_interval = 60

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose_estimation_params.process(frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        neck = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]

        shoulder_hip_vector = np.array([shoulder[0]-hip[0], shoulder[1]-hip[1]])
        neck_shoulder_vector = np.array([neck[0]-shoulder[0], neck[1]-shoulder[1]])
        angle = np.degrees(np.arccos(np.dot(shoulder_hip_vector, neck_shoulder_vector)/(np.linalg.norm(shoulder_hip_vector)*np.linalg.norm(neck_shoulder_vector))))

        current_time = time.time()
        if current_time - last_logged_time> log_interval:
            with open('posture_data.csv', 'a', newline='') as f:
                writer=csv.writer(f)
                writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), angle])
            last_logged_time = current_time

        if angle>35 and current_time - last_alert_time >alert_interval:
            engine.say(text)
            engine.runAndWait()
            last_alert_time= current_time

        if angle>35:
            cv2.rectangle(frame, (0,0),(frame.shape[1], frame.shape[0]),(0,0,255),20)

        cv2.line(frame, (int(hip[0]*frame.shape[1]), int(hip[1]*frame.shape[0])),
                 (int(shoulder[0]*frame.shape[1]), int(shoulder[1]*frame.shape[0])), (0,255,0), 2)
        cv2.line(frame, (int(shoulder[0] * frame.shape[1]), int(shoulder[1] * frame.shape[0])),
                 (int(neck[0] * frame.shape[1]), int(neck[1] * frame.shape[0])), (0, 255, 0), 2)

        cv2.circle(frame, (int(shoulder[0]*frame.shape[1]), int(shoulder[1]*frame.shape[0])), 5,(0,255,0),-1)
        cv2.rectangle(frame, (10, 10), (600, 80), (0, 0, 0), -1)

        cv2.putText(frame, f'Angle: {angle:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3,
                    cv2.LINE_AA)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()