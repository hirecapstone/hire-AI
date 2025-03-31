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

def detect_smile_change(face_avg, baseline):
    if len(face_avg) < 300 or baseline is None:
        return "중립"
    smile_score = face_avg[13][1] - (face_avg[61][1] + face_avg[291][1]) / 2
    baseline_score = baseline[13][1] - (baseline[61][1] + baseline[291][1]) / 2
    return "웃음" if smile_score - baseline_score > 0.005 else "중립"

def detect_posture_change(pose_avg, baseline):
    if len(pose_avg) < 13 or baseline is None:
        return "정자세입니다"
    ls, rs = pose_avg[11], pose_avg[12]
    base_ls, base_rs = baseline[11], baseline[12]

    dy = (ls[1] - rs[1]) - (base_ls[1] - base_rs[1])
    dz = ((ls[2] + rs[2]) / 2) - ((base_ls[2] + base_rs[2]) / 2)

    if dy > 0.04:
        return "왼쪽으로 기울어졌습니다"
    elif dy < -0.04:
        return "오른쪽으로 기울어졌습니다"
    elif dz < -0.07:
        return "몸이 앞으로 숙여졌습니다"
    elif dz > 0.07:
        return "몸이 뒤로 젖혀졌습니다"
    return "정자세입니다"

def detect_gaze_direction(face_avg, baseline_eye_diff):
    if len(face_avg) < 468:
        return True, baseline_eye_diff
    left_x, right_x = face_avg[33][0], face_avg[263][0]
    eye_diff = abs(left_x - right_x)
    if baseline_eye_diff is None:
        return True, eye_diff
    return abs(eye_diff - baseline_eye_diff) < 0.02, baseline_eye_diff

def detect_iris_direction(landmarks):
    try:
        left_iris_x = landmarks[468][0]
        left_center_x = (landmarks[33][0] + landmarks[133][0]) / 2
        right_iris_x = landmarks[473][0]
        right_center_x = (landmarks[362][0] + landmarks[263][0]) / 2
        avg_offset = ((left_iris_x - left_center_x) + (right_iris_x - right_center_x)) / 2
        return abs(avg_offset) < 0.02
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
    face_intervals, posture_intervals, not_looking_intervals = [], [], []

    face_state, face_start = "중립", None
    posture_state, posture_start = "정자세입니다", None
    posture_change_count = 0
    not_looking_count = 0

    face_baseline, pose_baseline, baseline_eye_diff = None, None, None
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

            if current_sec < 2:
                if face_avg is not None:
                    face_baseline = face_avg
                    _, baseline_eye_diff = detect_gaze_direction(face_avg, None)
                if pose_avg is not None:
                    pose_baseline = pose_avg
            else:
                if face_avg is not None:
                    new_face_state = detect_smile_change(face_avg, face_baseline)
                    if new_face_state != face_state:
                        if face_state != "중립" and face_start is not None:
                            face_intervals.append((face_start, current_sec, face_state))
                        face_start = current_sec if new_face_state != "중립" else None
                        face_state = new_face_state

                    is_looking, baseline_eye_diff = detect_gaze_direction(face_avg, baseline_eye_diff)
                    iris_check = detect_iris_direction(face_avg)
                    if not is_looking or not iris_check:
                        not_looking_intervals.append(current_sec)
                        not_looking_count += 1

                if pose_avg is not None:
                    new_posture = detect_posture_change(pose_avg, pose_baseline)
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

    elapsed = time.time() - start_time
    minutes, seconds = int(elapsed // 60), int(elapsed % 60)

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

    print(f"\n⏱️ 분석 소요 시간: {minutes}분 {seconds}초 ({elapsed:.2f}초)")

# ===== 실행 =====
if __name__ == "__main__":
    video_path = download_latest_video_from_firebase()
    if video_path:
        analyze_video(video_path)
