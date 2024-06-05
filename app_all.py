import streamlit as st
import mediapipe as mp
import pandas as pd
import cv2
import numpy as np
import joblib
import tempfile

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 부위 번호
LEFT_ANKLE = 27
LEFT_KNEE = 25
LEFT_HIP = 23
LEFT_EAR = 7
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

knee_model_path = "knee_model.pkl"  # 무릎 부위 모델
knee_model = joblib.load(knee_model_path)
spine_model_path = 'spine_model.pkl'  # 척추 모델
spine_model = joblib.load(spine_model_path)
arm_model_path = 'arm_model.pkl'  # 팔 모델
arm_model = joblib.load(arm_model_path)

expected_features_knee = knee_model.feature_names_in_
expected_features_spine = spine_model.feature_names_in_
expected_features_arm = arm_model.feature_names_in_

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

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

st.title("런닝 전문가")

st.subheader('당신의 런닝 자세를 전문가 수준으로')

uploaded_file = st.file_uploader("비디오를 선택하세요", type=["mp4", "avi", "mov", "mkv"])

st.text('정확한 분석을 위해 왼쪽에서 찍은 사진을 올려주세요!')

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    interval = 5  # 5초 간격 좌표 추출
    frame_rate = cap.get(cv2.CAP_PROP_FPS) 
    frame_interval = int(frame_rate * interval)  
    coordinates = []


    info_placeholder = st.empty()#
    analyze_button = st.button('분석하기')
    info_placeholder.info('분석하는데 시간이 좀 걸려요. 버튼 누르고 잠시 기다려 주세요.') ##


    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                neck_angle = calculate_angle([landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y],
                                             [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                                             [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(image, f'Elbow Angle: {int(left_elbow_angle)}', 
                            tuple(np.multiply(left_elbow, [image.shape[1], image.shape[0]]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Knee Angle: {int(left_knee_angle)}', 
                            tuple(np.multiply(left_knee, [image.shape[1], image.shape[0]]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Neck Angle: {int(neck_angle)}', 
                            (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if len(coordinates) == 0 or cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval == 0:
                    coordinates.append({
                        "frame": cap.get(cv2.CAP_PROP_POS_FRAMES),
                        "left_ankle_x": landmarks[LEFT_ANKLE].x,
                        "left_ankle_y": landmarks[LEFT_ANKLE].y,
                        "left_knee_x": landmarks[LEFT_KNEE].x,
                        "left_knee_y": landmarks[LEFT_KNEE].y,
                        "left_hip_x": landmarks[LEFT_HIP].x,
                        "left_hip_y": landmarks[LEFT_HIP].y,
                        "left_ear_x": landmarks[LEFT_EAR].x,
                        "left_ear_y": landmarks[LEFT_EAR].y,
                        "left_shoulder_x": landmarks[LEFT_SHOULDER].x,
                        "left_shoulder_y": landmarks[LEFT_SHOULDER].y,
                        "left_elbow_x": landmarks[LEFT_ELBOW].x,
                        "left_elbow_y": landmarks[LEFT_ELBOW].y,
                        "left_wrist_x": landmarks[LEFT_WRIST].x,
                        "left_wrist_y": landmarks[LEFT_WRIST].y
                    })

            stframe.image(image, channels='BGR')

    cap.release()
    tfile.close()

    if analyze_button:
        # 무릎 모델 predict
        results_knee = predict(coordinates, knee_model, expected_features_knee)
        knee_issues = results_knee['prediction'].isin([0]).any()

        # 척추 모델 predict
        results_spine = predict(coordinates, spine_model, expected_features_spine)
        spine_issues = results_spine['prediction'].isin([0]).any()

        # 팔 모델 predict
        results_arm = predict(coordinates, arm_model, expected_features_arm)
        arm_issues = results_arm['prediction'].isin([0]).any()

        info_placeholder.empty()


        if spine_issues:
            st.error('등과 후두부가 일직선이 되도록 해야 해요. 어깨와 가슴을 펴세요.')
        if arm_issues:
            st.error('팔꿈치의 각도는 고정시키지 않고 탄력성을 줘야 해요. 팔을 뒤로 친다는 생각으로 달려 보세요.')
        if knee_issues:
            st.error('무릎을 너무 많이 올리면 에너지 소모가 심해요. 적당히 편안한 보폭을 유지하고, 착지 시 무릎을 살짝 굽혀서 충격을 최소화 하세요')
        if not (spine_issues or arm_issues or knee_issues):
            st.success('올바른 자세로 잘 뛰고 있군요! 아주 좋습니다')