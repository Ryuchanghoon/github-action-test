import cv2
import mediapipe as mp
import numpy as np

# 각도 계산 함수
def calculate_angle(landmark1, landmark2):
    dy = landmark2[1] - landmark1[1]
    dx = landmark2[0] - landmark1[0]
    angle_radians = np.arctan2(dy, dx)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# 좌표 추출 함수
def extract_coordinates(results):
    landmarks = results.pose_landmarks.landmark
    coordinates = {
        "left_ear": [landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].y],
        "left_shoulder": [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
    }
    return coordinates

# 비디오 처리 함수
def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ["Error: Cannot open video."]

    extracted_coordinates = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            coordinates = extract_coordinates(results)
            extracted_coordinates.append(coordinates)

    cap.release()
    return extracted_coordinates


def test_process_video():
    test_video_path = "path/to/test_video.mp4"
    results = process_video(test_video_path)

if __name__ == "__main__":
    test_process_video()