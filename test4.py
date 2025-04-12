import cv2
import mediapipe as mp
import time

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(0)

# 기준값 저장용
baseline_mouth_open = []
baseline_iris_ratios = []

baseline_collected = False
baseline_start_time = time.time()

last_print_time = time.time()

# 입 벌림 거리 계산
def get_mouth_open(face_landmarks):
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    return abs(top_lip.y - bottom_lip.y)

# 눈동자 상대 위치 비율 계산
def get_normalized_iris_ratio(face_landmarks):
    # 왼쪽 눈 기준
    left_eye_left = face_landmarks.landmark[33]
    left_eye_right = face_landmarks.landmark[133]
    left_iris = face_landmarks.landmark[468]

    eye_width = left_eye_right.x - left_eye_left.x
    if eye_width == 0:
        return 0.5  # 나누기 0 방지

    iris_ratio = (left_iris.x - left_eye_left.x) / eye_width
    return iris_ratio

# 표정 분석
def analyze_expression(mouth_open, baseline):
    threshold = baseline + 0.01  # 기준보다 0.01 이상 크면 웃음
    return "웃음" if mouth_open > threshold else "무표정"

# 시선 분석
def analyze_gaze(iris_ratio, baseline):
    threshold = 0.1  # 기준보다 10% 이상 차이 나면 정면 아님
    diff = abs(iris_ratio - baseline)
    return "정면" if diff < threshold else "정면 아님"

print("👀 기준 수집 중입니다. 2초간 정면을 보고 입을 다문 상태로 있어주세요...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    expression = "감지 안됨"
    gaze = "감지 안됨"

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        mouth_open = get_mouth_open(face_landmarks)
        iris_ratio = get_normalized_iris_ratio(face_landmarks)

        # 기준값 수집 (2초 동안)
        if not baseline_collected:
            if time.time() - baseline_start_time < 2:
                baseline_mouth_open.append(mouth_open)
                baseline_iris_ratios.append(iris_ratio)
            else:
                # 평균값 계산
                mouth_baseline = sum(baseline_mouth_open) / len(baseline_mouth_open)
                iris_baseline = sum(baseline_iris_ratios) / len(baseline_iris_ratios)
                baseline_collected = True
                print("✅ 기준값 수집 완료! 분석을 시작합니다.\n")
        else:
            # 분석 시작
            expression = analyze_expression(mouth_open, mouth_baseline)
            gaze = analyze_gaze(iris_ratio, iris_baseline)

    # 1초마다 출력
    current_time = time.time()
    if current_time - last_print_time >= 1 and baseline_collected:
        print(f"[표정: {expression}] [시선: {gaze}]")
        last_print_time = current_time

    cv2.imshow("Expression and Gaze", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
