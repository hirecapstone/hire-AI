from faster_whisper import WhisperModel
import os
import subprocess
import sys
import time
import warnings
import firebase_admin
from firebase_admin import credentials, firestore

# 경고 무시
warnings.filterwarnings("ignore")

# Firebase 초기화
firebase_key_path = os.path.join("firebase", "hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)

# Firestore 참조
db = firestore.client()

# 인자로 받은 mp4 파일명
if len(sys.argv) < 2:
    print("❌ mp4 파일명을 인자로 넘겨주세요.")
    sys.exit(1)

mp4_filename = sys.argv[1]
filename_wo_ext = os.path.splitext(mp4_filename)[0]
wav_filename = f"{filename_wo_ext}.wav"

# ffmpeg로 mp4 → wav 변환
subprocess.run(['ffmpeg', '-y', '-i', mp4_filename, wav_filename])

# faster-whisper 모델 실행
start_time = time.time()
model = WhisperModel("medium", compute_type="int8", device="cpu")
segments, info = model.transcribe(wav_filename, beam_size=1, language="ko")

print("📝 텍스트 변환 결과:")
transcribed_text = ""
for segment in segments:
    transcribed_text += segment.text + " "

# Firestore에 텍스트 저장
if transcribed_text:
    # interview_answers 컬렉션에 새로운 문서 생성
    doc_ref = db.collection('interview_answers').document()
    doc_ref.set({
        'text': transcribed_text
    
    })
    # print("✅ Firebase에 텍스트 저장 완료")

end_time = time.time()
print("⏱️ Whisper 소요 시간:", round(end_time - start_time, 2), "seconds")
