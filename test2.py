import firebase_admin
from firebase_admin import credentials, storage
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import sys

if len(sys.argv) < 2:
    print("❌ mp4 파일명을 인자로 넘겨주세요.")
    sys.exit(1)

video_path = sys.argv[1]

# ===== 분석 함수 =====
def extract_landmarks(results, face=False, pose=False):
    landmarks = []
    if face and results.multi_face_landmarks:
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    elif pose and results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    return landmarks

def calculate_smile_score(landmarks):
    try:
        left = np.array(landmarks[61])
        right = np.array(landmarks[291])
        mid = np.array(landmarks[13])  # 윗입술 중앙

        mouth_width = np.linalg.norm(right[:2] - left[:2])
        mouth_height = abs(mid[1] - (left[1] + right[1]) / 2)
        slope = (left[1] + right[1]) / 2 - mid[1]

        return (mouth_height + slope) / mouth_width
    except:
        return None

def calculate_eye_diff(landmarks):
    if len(landmarks) < 468:
        return None
    return abs(landmarks[33][0] - landmarks[263][0])

def calculate_posture_metrics(pose_lm):
    if len(pose_lm) < 13:
        return None, None
    ls = pose_lm[11]
    rs = pose_lm[12]
    shoulder_y_diff = ls[1] - rs[1]
    shoulder_z = (ls[2] + rs[2]) / 2
    return shoulder_y_diff, shoulder_z

def detect_iris_direction_refined(landmarks):
    try:
        left_iris_x = landmarks[468][0]
        left_inner = landmarks[133][0]
        left_outer = landmarks[33][0]
        left_width = abs(left_outer - left_inner)
        left_offset_ratio = abs(left_iris_x - (left_inner + left_outer) / 2) / left_width

        right_iris_x = landmarks[473][0]
        right_inner = landmarks[362][0]
        right_outer = landmarks[263][0]
        right_width = abs(right_outer - right_inner)
        right_offset_ratio = abs(right_iris_x - (right_inner + right_outer) / 2) / right_width

        avg_offset_ratio = (left_offset_ratio + right_offset_ratio) / 2
        return avg_offset_ratio < 0.01
    except:
        return True

# ===== 메인 분석 함수 =====
def analyze_video(video_path):
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    pose = mp_pose.Pose()
    face = mp_face_mesh.FaceMesh(refine_landmarks=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"🎞 FPS: {fps}, 전체 프레임: {frame_count}, 영상 길이: {duration:.2f}초")

    face_buffer, pose_buffer = [], []
    smile_scores, eye_diffs, posture_ys, posture_zs = [], [], [], []

    face_intervals, posture_intervals, not_looking_intervals = [], [], []
    face_state, face_start = "중립", None
    posture_state, posture_start = "정자세입니다", None
    posture_change_count = 0
    not_looking_count = 0

    current_sec, frame_idx = 0, 0
    start_time = time.time()

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
            face_valid = [np.array(f) for f in face_buffer if len(f) > 0]
            pose_valid = [np.array(p) for p in pose_buffer if len(p) > 0]
            face_avg = np.mean(face_valid, axis=0) if face_valid else None
            pose_avg = np.mean(pose_valid, axis=0) if pose_valid else None

            if current_sec < 5:
                if face_avg is not None:
                    smile = calculate_smile_score(face_avg)
                    eye = calculate_eye_diff(face_avg)
                    if smile is not None:
                        smile_scores.append(smile)
                    if eye is not None:
                        eye_diffs.append(eye)
                if pose_avg is not None:
                    y_diff, z_avg = calculate_posture_metrics(pose_avg)
                    if y_diff is not None:
                        posture_ys.append(y_diff)
                    if z_avg is not None:
                        posture_zs.append(z_avg)
            else:
                if face_avg is not None:
                    smile_score = calculate_smile_score(face_avg)
                    eye_diff = calculate_eye_diff(face_avg)
                    iris_check = detect_iris_direction_refined(face_avg)
                    baseline_smile = np.mean(smile_scores) if smile_scores else 0
                    baseline_eye = np.mean(eye_diffs) if eye_diffs else 0

                    # 디버깅 출력
                    print(f"[DEBUG] 초:{current_sec}, smile_score:{smile_score:.4f}, baseline:{baseline_smile:.4f}, diff:{smile_score - baseline_smile:.4f}")

                    new_face_state = "웃음" if smile_score - baseline_smile > 0.11 else "중립"
                    if new_face_state != face_state:
                        if face_state != "중립" and face_start is not None:
                            face_intervals.append((face_start, current_sec, face_state))
                        face_start = current_sec if new_face_state != "중립" else None
                        face_state = new_face_state

                    if eye_diff - baseline_eye > 0.02 or not iris_check:
                        not_looking_intervals.append(current_sec)
                        not_looking_count += 1

                if pose_avg is not None:
                    y_diff, z_avg = calculate_posture_metrics(pose_avg)
                    base_y = np.mean(posture_ys) if posture_ys else 0
                    base_z = np.mean(posture_zs) if posture_zs else 0
                    dy = y_diff - base_y
                    dz = z_avg - base_z

                    if dy > 0.04:
                        new_posture = "왼쪽으로 기울어졌습니다"
                    elif dy < -0.04:
                        new_posture = "오른쪽으로 기울어졌습니다"
                    elif dz < -0.1:
                        new_posture = "몸이 앞으로 숙여졌습니다"
                    elif dz > 0.1:
                        new_posture = "몸이 뒤로 젖혀졌습니다"
                    else:
                        new_posture = "정자세입니다"

                    if new_posture != posture_state:
                        if posture_state != "정자세입니다" and posture_start is not None:
                            posture_intervals.append((posture_start, current_sec, posture_state))
                            posture_change_count += 1
                        posture_start = current_sec if new_posture != "정자세입니다" else None
                        posture_state = new_posture

            face_buffer.clear()
            pose_buffer.clear()
            current_sec += 1

    cap.release()
    if face_state != "중립" and face_start is not None:
        face_intervals.append((face_start, current_sec, face_state))
    if posture_state != "정자세입니다" and posture_start is not None:
        posture_intervals.append((posture_start, current_sec, posture_state))
        posture_change_count += 1

    end_time = time.time()

    # ===== 결과 출력 =====
    print("\n🙂 얼굴 표정 분석 결과:")
    if face_intervals:
        for s, e, state in face_intervals:
            print(f" - {s}초 ~ {e}초 사이에 {state} 표정을 지었습니다.")
    else:
        print(" - 감지된 표정 변화가 없습니다.")

    print("\n🕺 자세 분석 결과:")
    if posture_intervals:
        for s, e, state in posture_intervals:
            print(f" - {s}초 ~ {e}초 사이에 자세가 {state}")
    else:
        print(" - 감지된 자세 변화가 없습니다.")

    print("\n👁️ 시선 분석 결과:")
    if not_looking_intervals:
        for sec in not_looking_intervals:
            print(f" - {sec}초에 정면을 바라보지 않았습니다.")
    else:
        print(" - 대부분 정면을 바라보았습니다.")

    smile_duration = sum(e - s for s, e, st in face_intervals if st == "웃음")
    neutral_duration = duration - smile_duration

    print("\n📝 표정 피드백:")
    if smile_duration > neutral_duration:
        print(f" - 웃는 표정이 더 많았습니다 ({smile_duration:.1f}초). 잘하고 있습니다! 😄")
    elif smile_duration == 0:
        print(" - 웃는 표정이 한 번도 감지되지 않았습니다. 좀 더 미소를 지어보세요. 🙂")
    else:
        print(f" - 무표정이 더 많았습니다 ({neutral_duration:.1f}초). 좀 더 웃는 표정을 유지해보세요. 🙂")

    print("\n📌 자세 피드백:")
    if posture_change_count == 0:
        print(" - 자세 변화가 거의 없었습니다. 매우 안정된 자세를 유지했습니다. 👍")
    elif posture_change_count <= 2:
        print(f" - 자세 변화가 {posture_change_count}회 감지되었습니다. 비교적 안정적인 자세를 유지했습니다.")
    else:
        print(f" - 자세 변화가 자주 발생했습니다 (총 {posture_change_count}회). 안정된 자세를 유지해보세요. 🙂")

    print("\n👁️ 시선 피드백:")
    if not_looking_count >= 5:
        print(f" - 정면을 바라보지 않은 경우가 {not_looking_count}회 감지되었습니다. 카메라를 더 응시해주세요. 👀")
    else:
        print(f" - 대부분 정면을 잘 응시하였습니다! ({not_looking_count}회만 감지됨) 👍")

    print(f"\n⏱️ Mediapipe 소요 시간: {round(end_time - start_time, 2)} seconds")

# 실행
if __name__ == "__main__":
    analyze_video(video_path)
