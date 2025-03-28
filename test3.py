import os
from dotenv import load_dotenv
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, firestore
import time

#환경 변수 로드 및 OpenAI API 설정
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Firebase 초기화
cred = credentials.Certificate("firebase/hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json")  # Firebase 서비스 계정 키 파일 경로
firebase_admin.initialize_app(cred)
db = firestore.client()

#GPT로 면접 질문 생성 함수
def generate_questions(field="기업 - IT", num_questions=6):
    prompt = f"""
'{field}' 분야의 면접에서 사용할 수 있는 실무 중심 질문 {num_questions}개를 생성해줘.
질문은 구체적이고 실제 면접에서 사용 가능한 수준으로 작성해줘.
"""
    start_time = time.time()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 채용 담당자입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ 질문 생성 완료! 소요 시간: {elapsed_time:.2f}초")
    content = response.choices[0].message.content.strip()

    #(번호 제거 + 리스트 형태로 변환)
    questions = []
    for line in content.split('\n'):
        line = line.strip()
        if line and any(line.startswith(f"{i}.") for i in range(1, num_questions + 5)):  # 유연한 범위
            question = line.split('.', 1)[1].strip()
            questions.append(question)

    return questions[:num_questions]

#Firestore에 질문 저장
def save_questions_to_firebase(field, questions):
    doc_ref = db.collection("interview_questions").document()
    doc_ref.set({
        "field": field,
        "questions": questions
    })
    print(f"\n✅ {len(questions)}개의 질문이 Firebase에 저장되었습니다. (분야: {field})")

#(선택 분야가 없으므로 기본값 사용)
if __name__ == "__main__":
    default_field = "기업 - IT"
    questions = generate_questions(field=default_field)

    # 생성된 질문 출력
    print(f"\n📋 생성된 '{default_field}' 분야 면접 질문:")
    for idx, q in enumerate(questions, 1):
        print(f"{idx}. {q}")

    # Firebase에 저장
    save_questions_to_firebase(default_field, questions)
