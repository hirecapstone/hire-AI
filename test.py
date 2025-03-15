import firebase_admin
from firebase_admin import credentials, storage

# 서비스 계정 키 로드
cred = credentials.Certificate("firebase/hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json")

# Firebase 초기화 (storageBucket 값을 Firebase 콘솔에서 복사한 URL로 설정)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'hire-ai-a11ed.firebasestorage.app'
})

# Firebase Storage 접근
bucket = storage.bucket()

print(f"✅ Firebase Storage 연결 성공: {bucket.name}")

# Firebase Storage에 저장된 모든 파일 가져오기
def list_files():
    blobs = bucket.list_blobs()  # Storage 내 모든 파일 가져오기
    files = [blob.name for blob in blobs]  # 파일 이름 목록 생성
    return files

# 실행
files = list_files()
print("📂 Firebase Storage 파일 목록:")
for file in files:
    print(f" - {file}")
