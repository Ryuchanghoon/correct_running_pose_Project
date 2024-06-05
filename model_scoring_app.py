import streamlit as st
import mediapipe as mp
import pandas as pd
import cv2
#import numpy as np
import joblib


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 랜드마크
LEFT_ANKLE = 27
LEFT_KNEE = 25
LEFT_HIP = 23

LEFT_EAR = 7
LEFT_SHOULDER = 11

LEFT_ELBOW = 13
LEFT_WRIST = 15


knee_model_path = "knee_model.pkl" # 무릎 부위 모델
knee_model = joblib.load(knee_model_path)

spine_model_path = 'spine_model.pkl'
spine_model = joblib.load(spine_model_path)

arm_model_path = 'arm_model.pkl'
arm_model = joblib.load(arm_model_path)



expected_features_knee = knee_model.feature_names_in_
expected_features_spine = spine_model.feature_names_in_
expected_features_arm = arm_model.feature_names_in_


def process_video(video_path, interval):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * interval)
    frame_number = 0
    data = []

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % frame_interval == 0:
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    left_ankle = results.pose_landmarks.landmark[LEFT_ANKLE]
                    left_knee = results.pose_landmarks.landmark[LEFT_KNEE]
                    left_hip = results.pose_landmarks.landmark[LEFT_HIP]
                    left_ear = results.pose_landmarks.landmark[LEFT_EAR]
                    left_shoulder = results.pose_landmarks.landmark[LEFT_SHOULDER]
                    left_elbow = results.pose_landmarks.landmark[LEFT_ELBOW]
                    left_wrist = results.pose_landmarks.landmark[LEFT_WRIST]
                    coordinates = {
                        "frame": frame_number,
                        "left_ankle_x": left_ankle.x,
                        "left_ankle_y": left_ankle.y,
                        "left_knee_x": left_knee.x,
                        "left_knee_y": left_knee.y,
                        "left_hip_x": left_hip.x,
                        "left_hip_y": left_hip.y,
                        "left_ear_x": left_ear.x,
                        "left_ear_y": left_ear.y,
                        "left_shoulder_x": left_shoulder.x,
                        "left_shoulder_y": left_shoulder.y,
                        "left_elbow_x": left_elbow.x,
                        "left_elbow_y": left_elbow.y,
                        "left_wrist_x": left_wrist.x,
                        "left_wrist_y": left_wrist.y
                    }
                    data.append(coordinates)
            frame_number += 1

    cap.release()
    return data


def predict(data, model, expected_features):
    df = pd.DataFrame(data)
    features = df[expected_features]
    predictions = model.predict(features)
    df['prediction'] = predictions
    return df

st.title("자세 예측")

uploaded_file = st.file_uploader("비디오를 선택하세요.", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file is not None:
    video_path = f"temp_video.{uploaded_file.name.split('.')[-1]}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    interval = 5  # 5초 간격 좌표 추출
    coordinates = process_video(video_path, interval)
    if coordinates:
        st.write("추출된 좌표 수:", len(coordinates))
        st.write(coordinates)
        
        # Knee model prediction
        results_knee = predict(coordinates, knee_model, expected_features_knee)
        st.write("무릎 모델 예측 결과:")
        for index, row in results_knee.iterrows():
            st.write(f"Frame: {row['frame']}, Prediction: {row['prediction']}")

        # Spine model prediction
        results_spine = predict(coordinates, spine_model, expected_features_spine)
        st.write("척추 모델 예측 결과:")
        for index, row in results_spine.iterrows():
            st.write(f"Frame: {row['frame']}, Prediction: {row['prediction']}")

        # Arm model prediction
        results_arm = predict(coordinates, arm_model, expected_features_arm)
        st.write("팔 모델 예측 결과:")
        for index, row in results_arm.iterrows():
            st.write(f"Frame: {row['frame']}, Prediction: {row['prediction']}")

    else:
        st.write("포즈 랜드마크 추출 안됨")