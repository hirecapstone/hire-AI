import os
import time
import warnings
import re
import json
import glob
import subprocess

import openai
from openai import OpenAI
from firebase_admin import credentials, firestore, initialize_app, storage
from flask import Flask, request, jsonify
from google.api_core.exceptions import AlreadyExists

# ————— 설정 —————
MAX_BYTES = 25 * 1024 * 1024  # Whisper API 최대 파일 크기 (25MB)
SEGMENT_SECONDS = 300
# 말 빠르기 분류 기준 (WPM)
SLOW_WPM = 80.0
FAST_WPM = 120.0
warnings.filterwarnings("ignore")

# OpenAI 키 (환경변수에 미리 설정해주세요)
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

# Firebase 초기화
cred_path = os.path.join(
    os.path.dirname(__file__),
    "firebase/hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json"
)
initialize_app(
    credentials.Certificate(cred_path),
    {'storageBucket': 'hire-ai-a11ed.firebasestorage.app'}
)

db     = firestore.client()
bucket = storage.bucket()
app    = Flask(__name__)

def classify_speed(wpm: float, slow_wpm: float = SLOW_WPM, fast_wpm: float = FAST_WPM) -> str:
    """
    WPM 기준으로 말 빠르기 라벨링:
      wpm == 0                 → 'no_speech'  (발화 없음)
      wpm < slow_wpm           → 'slow'
      slow_wpm <= wpm <= fast_wpm → 'normal'
      wpm > fast_wpm           → 'fast'
    """
    if wpm == 0:
        return "no_speech"
    if wpm < slow_wpm:
        return "slow"
    elif wpm > fast_wpm:
        return "fast"
    else:
        return "normal"
    
def _split_audio(path: str) -> list[str]:
    """
    FFmpeg를 사용해 WAV 파일을 여러 개의 세그먼트로 분할합니다.
    """
    base = os.path.splitext(path)[0]
    pattern = f"{base}_seg%03d.wav"
    subprocess.run([
        'ffmpeg', '-y', '-i', path,
        '-ar', '16000', '-ac', '1',
        '-f', 'segment', '-segment_time', str(SEGMENT_SECONDS), pattern
    ], check=True)
    return sorted(glob.glob(f"{base}_seg*.wav"))


def transcribe_video(path: str,silence_thresh: float = 0.6) -> tuple[str, float]:
    """
    Whisper API 전사용. 파일이 25MB를 넘으면 오디오로 변환 후 분할하여 전사.
    """
    parts = [path]
    if os.path.getsize(path) > MAX_BYTES:
        wav_path = path.replace('.mp4', '.wav')
        subprocess.run(['ffmpeg', '-y', '-i', path, wav_path], check=True)
        parts = _split_audio(wav_path)

    all_texts      = []
    total_duration = 0.0  # seconds
    total_words    = 0

    for part in parts:
        with open(part, 'rb') as f:
            resp = client.audio.transcriptions.create(
                file=f,
                model='whisper-1',
                response_format='verbose_json', 
                temperature=0,                   
                language='ko'
            )
        segments = [
            seg for seg in resp.segments
            if getattr(seg, "no_speech_prob", 1.0) < silence_thresh
        ]
        # 전사 텍스트 모으기
        text = "".join(seg.text for seg in segments)
        all_texts.append(text)
        # 재생 시간 및 단어수 합산
        total_duration += sum((seg.end - seg.start) for seg in segments)
        total_words    += sum(len(seg.text.split()) for seg in segments)
    transcription = ' '.join(all_texts)
    wpm = (total_words / total_duration * 60) if total_duration > 0 else 0.0
    return transcription, wpm


def process_videos(session_id: str):
    """
    1) videos/{session_id}/*.mp4 9개 → Whisper로 자막 뽑아 interview_answers에 저장
    2) interview_questions에서 질문 9개 읽어와 ChatGPT 피드백+점수 → interview_feedback에 저장
    각 단계별 소요시간 로그 출력.
    """
    t0_total = time.time()
    print(f"[{session_id}] ▶️ 배치 처리 시작", flush=True)

    # 1) Whisper 처리 & interview_answers 저장
    t0_wh = time.time()
    prefix = f"videos/{session_id}/"
    blobs  = list(bucket.list_blobs(prefix=prefix))
    videos = sorted([b for b in blobs if b.name.lower().endswith(".mp4")], key=lambda b: b.name)

    transcripts = []
    speed_labels = []

    for idx, blob in enumerate(videos, start=1):
        name      = os.path.basename(blob.name)
        local_mp4 = f"/tmp/{name}"

        dl0 = time.time()
        blob.download_to_filename(local_mp4)
        print(f"[{session_id}] ({idx}/{len(videos)}) 다운로드 완료 (⏱ {time.time()-dl0:.1f}s)", flush=True)

        wh0 = time.time()
        try:
            text, wpm = transcribe_video(local_mp4)
            speed = classify_speed(wpm)
        except Exception as e:
            print(f"[{session_id}] ({idx}) 전사 실패: {e}", flush=True)
            text = ""
            speed = "no_speech"
            wpm    = 0.0
        transcripts.append(text)
        speed_labels.append(speed)
        print(f"[{session_id}] ({idx}/{len(videos)}) Whisper 분석 완료 (⏱ {time.time()-wh0:.1f}s)", flush=True)
    print(f"[{session_id}] 📝 Whisper 전체 처리 시간: {time.time()-t0_wh:.1f}s", flush=True)

    db.collection('interview_answers').document(session_id).set({
        'text': transcripts
    })
    print(f"[{session_id}] ✅ interview_answers 저장 완료", flush=True)

    # 2) ChatGPT 피드백+점수 생성 & interview_feedback 저장
    t0_q = time.time()
    q_doc = db.collection('interview_questions').document(session_id).get()
    if q_doc.exists:
        q_data = q_doc.to_dict()
        questions = [] + q_data.get('questions', [])
    else:
        questions = []
    print(f"[{session_id}] ℹ️ 질문 로딩 완료 (⏱ {time.time()-t0_q:.1f}s)", flush=True)

    feedbacks = []
    total_scores = []
    scores_details = []
    t0_fb = time.time()

    for idx, (q, a) in enumerate(zip(questions, transcripts), start=1):
        fb0 = time.time()
        prompt = (
            f"질문: {q}\n"
            f"답변: {a}\n\n"
            "아래 형식(JSON)으로만 응답해 주세요:\n"
            "1) feedback: 한두 문장으로 간단한 피드백 작성\n"
            "2) scores: 다섯 가지 기준(Relevance, Completeness, Clarity, Logic, Length)에 대해 **0,1,2,3,4,5**의 정수만 사용하여 채점 (true/false 사용 금지)\n"
            "   **0.0에서 5.0 사이 실수**로 채점 (소수점 첫째 자리까지)\n"
            "   - 예: 0.0, 1.5, 3.2, 4.0, 5.0\n"
            "3) total: scores 값들의 평균을 소수점 둘째 자리까지 반올림\n\n"
            # 예시 JSON
            '{"feedback":"답변이 전반적으로 좋으나, 구체적인 예시가 부족합니다.",'
            '"scores":{"relevance":4.2,"completeness":3.5,"clarity":4.8,"logic":4.0,"length":2.7},'
            '"total":3.44}'
        )
        
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role":"system", "content":"너는 공정한 면접관입니다."},
                {"role":"user",   "content": prompt}
            ],
            temperature=0,
            max_tokens=256
        )
        
        raw = resp.choices[0].message.content
        print(f"[DEBUG][{session_id}] Raw GPT response (1st):\n{raw}", flush=True)
        data = json.loads(raw)

        feedbacks.append(data.get("feedback", ""))
        total_scores.append(float(data.get("total", 0.0)))

        detail = data.get("scores", {})
        detail = {k: round(float(v), 1) for k, v in data["scores"].items()}
        scores_details.append(detail)

        print(f"[{session_id}] ({idx}/{len(questions)}) 피드백+점수 생성 완료 (⏱ {time.time()-fb0:.1f}s)", flush=True)

    print(f"[{session_id}] 💬 피드백+점수 전체 생성 시간: {time.time()-t0_fb:.1f}s", flush=True)

    # 평균 계산
    avg_total = round(sum(total_scores) / len(total_scores), 2) if total_scores else 0
    avg_per_criterion = {}
    if scores_details:
        for key in scores_details[0].keys():
            avg_per_criterion[key] = round(
                sum(d[key] for d in scores_details) / len(scores_details), 2
            )

    # Firestore에 피드백과 점수 저장
    db.collection('interview_feedback').document(session_id).set({
        'feedbacks': feedbacks,
        'total_scores': total_scores,
        'scores_details': scores_details,
        'avg_total_score': avg_total,
        'avg_scores_per_criterion': avg_per_criterion,
        'speed': speed_labels
    })
    print(f"[{session_id}] ✅ interview_feedback 저장 완료 (평균총점={avg_total}, 기준별평균={avg_per_criterion})", flush=True)

    print(f"[{session_id}] 🎉 배치 처리 총 소요시간: {time.time()-t0_total:.1f}s", flush=True)

@app.route('/', methods=['POST'])
def handle_event():
    # Cloud Storage finalized 이벤트만 처리
    ce = request.headers.get('ce-type') or request.headers.get('Ce-Type')
    if ce != 'google.cloud.storage.object.v1.finalized':
        return jsonify({"status":"ignored"}), 200

    payload = request.get_json(silent=True) or {}
    data    = payload.get('data', payload)
    gen     = data.get('metageneration')
    name    = data.get('name')

    # 첫번째 메타젠이 아니거나 이름이 없으면 무시
    if str(gen) != "1" or not name:
        return jsonify({"status":"ignored"}), 200

    # videos/{session_id}/{session_id}_qN.mp4 패턴만
    if not re.fullmatch(r"videos/[^/]+/[^/]+_q\d+\.mp4", name):
        return jsonify({"status":"ignored"}), 200

    session_id = name.split('/', 2)[1]

    # Firestore에서 질문 개수 조회
    q_doc = db.collection('interview_questions').document(session_id).get()
    questions = q_doc.to_dict().get("questions", []) if q_doc.exists else []
    expected_videos = len(questions)

    # 업로드된 동영상 개수 확인
    prefix = f"videos/{session_id}/"
    blobs  = [b for b in bucket.list_blobs(prefix=prefix)
              if b.name.lower().endswith(".mp4")]
    if len(blobs) < expected_videos:
        print(f"[{session_id}] 아직 업로드 {len(blobs)}/{expected_videos}개 → waiting", flush=True)
        return jsonify({"status":"waiting"}), 200

    # 7. 최초 진입 시 락 생성 시도 (한 번만 성공)
    lock_ref = db.collection('batch_locks').document(session_id)
    try:
        lock_ref.create({'done': True})
    except AlreadyExists:
        print(f"[{session_id}] 이미 배치 처리됨 → skip", flush=True)
        return jsonify({"status":"skipped"}), 200

    # 8. 배치 처리 실행
    try:
        process_videos(session_id)
    except Exception as e:
        print(f"[{session_id}] 배치 처리 실패: {e}", flush=True)
        # 필요하다면 락을 지우고 재시도할 수 있습니다:
        # lock_ref.delete()
        return jsonify({"status":"error", "reason": str(e)}), 500



    return jsonify({"status":"done"}), 200
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
