import os
import warnings
import firebase_admin
from firebase_admin import credentials, firestore
from openai import OpenAI
from dotenv import load_dotenv
#0.006$ 비용 소요

# 경고 무시
warnings.filterwarnings("ignore")

# Firebase 초기화
firebase_key_path = os.path.join("firebase", "hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# .env 파일에서 API 키 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Firebase에서 가장 최근에 업데이트된 면접 질문 가져오기
def get_latest_interview_question():
    questions_ref = db.collection('interview_questions')
    query = questions_ref.order_by('__name__', direction=firestore.Query.DESCENDING).limit(1)  # 문서 ID를 기준으로 내림차순 정렬
    docs = query.stream()

    for doc in docs:
        return doc.to_dict() 
    return None  

# Firebase에서 가장 최근에 업데이트된 면접 답변 가져오기
def get_latest_interview_answers():
    answers_ref = db.collection('interview_answers')
    query = answers_ref.order_by('__name__', direction=firestore.Query.DESCENDING).limit(1)  # 문서 ID를 기준으로 내림차순 정렬
    docs = query.stream()

    # 가장 최근의 문서 가져오기
    for doc in docs:
        return doc.to_dict()  # 문서의 데이터 반환
    return None  # 답변이 없을 경우

# 면접 질문과 변환된 텍스트로 피드백 생성
def generate_feedback(transcribed_text, interview_question):
    prompt = f"면접 질문: {interview_question}\n답변: {transcribed_text}\n이 답변이 이 질문에 적합한지 피드백을 주세요."

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[{
            "role": "system", "content": "You are a helpful assistant."
        }, {
            "role": "user", "content": prompt
        }],
        max_tokens=300,
        temperature=0.7
    )

    feedback = response.choices[0].message.content.strip()
    return feedback

# 총평 생성
def generate_overall_feedback(answers, questions):
    prompt = "이 면접자의 총평을 작성해주세요. 전반적인 평가, 강점, 개선점을 포함해야 합니다. 각 질문에 대한 답변을 바탕으로 총평을 작성하세요. 답변은 500토큰 이내로 작성해주세요.\n"

    # 면접 질문과 답변을 포함한 텍스트 생성
    for i, question in enumerate(questions):
        if i < len(answers):  # 질문과 답변이 매칭될 수 있도록 범위 체크
            transcribed_text = answers[i]  # 해당 질문에 대한 답변
            prompt += f"질문: {question}\n답변: {transcribed_text}\n"

    # GPT에게 총평 요청
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[{
            "role": "system", "content": "You are a helpful assistant."
        }, {
            "role": "user", "content": prompt
        }],
        max_tokens=700,
        temperature=0.7
    )

    overall_feedback = response.choices[0].message.content.strip()
    return overall_feedback

# Firebase에 피드백과 총평 저장하기 위한 함수
def save_feedback_to_firestore(feedback_data, collection_name):
    doc_ref = db.collection(collection_name).document()  # 새 문서 생성
    doc_ref.set(feedback_data)  # 피드백 데이터를 Firestore에 저장

# 가장 최근 면접 질문과 답변 가져오기
latest_question_data = get_latest_interview_question()
latest_answer_data = get_latest_interview_answers()

if latest_question_data and latest_answer_data:
    # 기본 질문을 리스트에 추가
    basic_questions = [
        "자기소개를 해주세요.",
        "지원동기를 말해주세요.",
        "장단점을 말해주세요."
    ]

    # 면접 질문에서 'questions'만 추출
    questions = latest_question_data.get("questions", [])
    answers = latest_answer_data.get("text", [])

    # 기본 질문과 Firebase에서 가져온 질문을 합쳐 총 9개의 질문 만들기
    all_questions = basic_questions + questions  # 기본 질문 3개 + Firebase에서 가져온 6개 질문

    # 각 질문에 대해 피드백 생성하고 Firestore에 저장
    feedback_data = {}  # 피드백을 저장할 딕셔너리

    for i, question in enumerate(all_questions):
        if i < len(answers):  # 질문과 답변이 매칭될 수 있도록 범위 체크
            transcribed_text = answers[i]  # 해당 질문에 대한 답변
            feedback = generate_feedback(transcribed_text, question)  # 피드백 생성
            feedback_data[f"question_{i+1}"] = {
                "question": question,
                "answer": transcribed_text,
                "feedback": feedback
            }

    # 피드백 데이터를 Firestore에 저장
    save_feedback_to_firestore(feedback_data, "interview_feedbacks")

    # 면접자의 총평 생성
    overall_feedback = generate_overall_feedback(answers, all_questions)
    total_feedback_data = {
        "overall_feedback": overall_feedback
    }

    # 총평 데이터를 Firestore에 저장
    save_feedback_to_firestore(total_feedback_data, "interview_overall_feedback")

else:
    print("업데이트된 면접 질문이나 답변이 없습니다.")
