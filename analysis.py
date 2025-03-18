import whisper
import time

# 실행 시작 시간 기록
start_time = time.time()

# 모델 로드 및 음성 변환
model = whisper.load_model("base")
result = model.transcribe(r"C:\video\hong.mp4", fp16=False)

# 실행 종료 시간 기록
end_time = time.time()

# 실행 시간 출력
print("Transcription Time:", round(end_time - start_time, 2), "seconds")
print(result["text"])