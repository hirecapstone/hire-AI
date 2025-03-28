from faster_whisper import WhisperModel
import firebase_admin
from firebase_admin import credentials, storage
import os
import time
import subprocess
import warnings


# 경고 무시
warnings.filterwarnings("ignore")

# Firebase 인증키 상대경로 설정
firebase_key_path = os.path.join("firebase", "hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json")

# Firebase 초기화 (한 번만 실행됨)
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'hire-ai-a11ed.firebasestorage.app'
    })

# Firebase Storage 버킷
bucket = storage.bucket()

# 모든 mp4 파일 목록 가져오기
blobs = list(bucket.list_blobs())

# mp4 파일 중 가장 최근 파일 찾기
mp4_files = [b for b in blobs if b.name.endswith('.mp4')]
latest_file = max(mp4_files, key=lambda b: b.time_created)

print(f"가장 최근 업로드된 파일: {latest_file.name}")

# 로컬 파일 이름 설정
local_mp4 = "latest.mp4"
local_wav = "latest.wav"

# 기존 파일 제거
if os.path.exists(local_mp4): os.remove(local_mp4)
if os.path.exists(local_wav): os.remove(local_wav)

# mp4 다운로드
latest_file.download_to_filename(local_mp4)
print("Firebase에서 mp4 다운로드 완료")

# ffmpeg로 mp4 → wav 변환
subprocess.run(['ffmpeg', '-y', '-i', local_mp4, local_wav])

# faster-whisper 모델 실행
start_time = time.time()
model = WhisperModel("base", compute_type="int8", device="cpu")
segments, info = model.transcribe(local_wav, beam_size=5, language="ko")
end_time = time.time()

# 결과 출력
print("Transcription Time:", round(end_time - start_time, 2), "seconds")
print("텍스트 변환 결과:")
for segment in segments:
    print(segment.text)
