import firebase_admin
from firebase_admin import credentials, storage
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ===== Firebase Storage에서 가장 최근 영상 다운로드 =====
def download_latest_video_from_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase/hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'hire-ai-a11ed.firebasestorage.app'
        })

    bucket = storage.bucket()
    blobs = list(bucket.list_blobs())

    if not blobs:
        print("❌ Firebase Storage에 파일이 없습니다.")
        return None

    latest_blob = max(blobs, key=lambda b: b.updated)
    filename = latest_blob.name
    local_filename = os.path.basename(filename)

    latest_blob.download_to_filename(local_filename)
    print(f"✅ 최신 파일 '{filename}' 다운로드 완료 → {local_filename}")

    return local_filename

# ===== 랜드마크 분석 관련 함수 =====
def calc_landmark_diff(a, b):
    if len(a) != len(b):
        return float('inf')
    return np.mean([np.linalg.norm(np.array(a[i]) - np.array(b[i])) for i in range(len(a))])

def extract_landmarks(results, face=False, pose=False):
    landmarks = []
    if face and results.multi_face_landmarks:
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    elif pose and results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    return landmarks

def detect_smile(landmarks):
    if len(landmarks) < 300:
        return False
    left = landmarks[61][1]
    right = landmarks[291][1]
    mid = landmarks[13][1]
    smile_score = mid - (left + right) / 2
    return smile_score > 0.01

def detect_surprise(landmarks):
    if len(landmarks) < 300:
        return False
    top_lip = landmarks[13][1]
    bottom_lip = landmarks[14][1]
    mouth_open = abs(top_lip - bottom_lip)
    return mouth_open > 0.045

def detect_posture_tilt(pose_lm):
    if len(pose_lm) < 30:
        return "정자세입니다"
    ls = pose_lm[11]
    rs = pose_lm[12]
    lh = pose_lm[23]
    rh = pose_lm[24]

    shoulder_y_diff = ls[1] - rs[1]
    shoulder_z = (ls[2] + rs[2]) / 2
    hip_z = (lh[2] + rh[2]) / 2

    if shoulder_y_diff > 0.005:
        return "왼쪽으로 기울어졌습니다"
    elif shoulder_y_diff < -0.005:
        return "오른쪽으로 기울어졌습니다"
    elif shoulder_z < hip_z - 0.01:
        return "몸이 앞으로 숙여졌습니다"
    elif shoulder_z > hip_z + 0.01:
        return "몸이 뒤로 젖혀졌습니다"
    else:
        return "정자세입니다"

# ===== 영상 분석 메인 함수 =====
def analyze_video(video_path):
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    pose = mp_pose.Pose()
    face = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"🎞 FPS: {fps}, 전체 프레임: {frame_count}, 영상 길이: {duration:.2f}초")

    current_sec = 0
    frame_idx = 0

    face_comments = []
    pose_comments = []

    face_buffer = []
    pose_buffer = []

    prev_face_state = "중립"
    prev_posture = "정자세입니다"

    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = face.process(image_rgb)
        pose_result = pose.process(image_rgb)

        face_lm = extract_landmarks(face_result, face=True)
        pose_lm = extract_landmarks(pose_result, pose=True)

        face_buffer.append(face_lm)
        pose_buffer.append(pose_lm)

        frame_idx += 1
        if frame_idx >= (current_sec + 1) * fps:
            # 평균 계산 (주의: buffer가 비었거나 유효하지 않으면 건너뜀)
            face_valid = [np.array(f) for f in face_buffer if isinstance(f, list) and len(f) > 0]
            pose_valid = [np.array(p) for p in pose_buffer if isinstance(p, list) and len(p) > 0]

            face_avg = np.mean(face_valid, axis=0) if face_valid else None
            pose_avg = np.mean(pose_valid, axis=0) if pose_valid else None

            # 얼굴 표정 분석
            if isinstance(face_avg, np.ndarray) and face_avg.ndim == 2 and face_avg.shape[0] > 0:
                smile = detect_smile(face_avg)
                surprise = detect_surprise(face_avg)

                if smile and prev_face_state != "웃음":
                    face_comments.append(f"{current_sec}초 ~ {current_sec + 1}초 사이에 웃는 표정을 지었습니다.")
                    prev_face_state = "웃음"
                elif surprise and prev_face_state != "놀람":
                    face_comments.append(f"{current_sec}초 ~ {current_sec + 1}초 사이에 놀란 표정을 지었습니다.")
                    prev_face_state = "놀람"
                elif not smile and not surprise:
                    prev_face_state = "중립"

            # 자세 분석
            if isinstance(pose_avg, np.ndarray) and pose_avg.ndim == 2 and pose_avg.shape[0] > 0:
                posture = detect_posture_tilt(pose_avg)
                if posture != prev_posture and posture != "정자세입니다":
                    pose_comments.append(f"{current_sec}초 ~ {current_sec + 1}초 사이에 자세가 {posture}")
                prev_posture = posture

            face_buffer.clear()
            pose_buffer.clear()
            current_sec += 1

    cap.release()
    end = time.time()
    elapsed = end - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # 결과 출력
    print("\n🙂 얼굴 표정 분석 결과:")
    if face_comments:
        for c in face_comments:
            print(" -", c)
    else:
        print(" - 감지된 표정 변화가 없습니다.")

    print("\n🕺 자세 분석 결과:")
    if pose_comments:
        for c in pose_comments:
            print(" -", c)
    else:
        print(" - 감지된 자세 변화가 없습니다.")

    print(f"\n⏱️ 분석 소요 시간: {minutes}분 {seconds}초 ({elapsed:.2f}초)")

# ===== 실행 시작점 =====
if __name__ == "__main__":
    video_path = download_latest_video_from_firebase()
    if video_path:
        analyze_video(video_path)