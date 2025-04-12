import firebase_admin
from firebase_admin import credentials, storage
import os
import subprocess

# Firebase 초기화
firebase_key_path = os.path.join("firebase", "hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'hire-ai-a11ed.firebasestorage.app'
    })

# Firebase Storage에서 가장 최근 mp4 파일 다운로드a
bucket = storage.bucket()
blobs = list(bucket.list_blobs())
mp4_files = [b for b in blobs if b.name.endswith('.mp4')]

if not mp4_files:
    print("❌ mp4 파일이 없습니다.")
    exit()

latest_file = max(mp4_files, key=lambda b: b.time_created)
firebase_filename = os.path.basename(latest_file.name)

# 기존 파일 제거
if os.path.exists(firebase_filename): os.remove(firebase_filename)

# 다운로드
latest_file.download_to_filename(firebase_filename)
print(f"✅ Firebase에서 다운로드 완료: {firebase_filename}")



proc1 = subprocess.Popen(["python", "test.py", firebase_filename])
proc2 = subprocess.Popen(["python", "test2.py", firebase_filename])


proc1.wait()
proc2.wait()

