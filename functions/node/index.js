const functions = require('firebase-functions');
const admin = require('firebase-admin');
const { OpenAI } = require('openai');
const { defineSecret } = require('firebase-functions/params');

const OPENAI_API_KEY = defineSecret('OPENAI_API_KEY');

admin.initializeApp();
const db = admin.firestore();

exports.onSelectRoleCreated = functions
  .runWith({ secrets: ['OPENAI_API_KEY'] }) 
  .firestore
  .document('select_role/{docId}')
  .onCreate(async (snap, context) => {
    const data = snap.data();
    
    const sessionId = data.sessionId || ''; //sessionId추가


    const category = data.category || '';
    const job = data.job || '';
    const achievements = data.achievements || '';
    const career = data.career || '';
    const certificates = data.certificates || '';
    const major = data.major || '';
    const projects = data.projects || '';
    const roles = data.roles || '';
    const staticQuestions = [
      '자기소개 해주세요',
      '지원 동기는 무엇인가요?',
      '본인의 장단점을 말해주세요.'
    ];
    
    let fieldDescription = `${category} ${job}`;

    if (achievements) {
      fieldDescription += `, 성과: ${achievements}`;
    }
    if (certificates) {
      fieldDescription += `, 자격증: ${certificates}`;
    }
    if (career) {
      fieldDescription += `, 경력: ${career}`;
    }
    if (major) {
      fieldDescription += `, 전공: ${major}`;
    }
    if (projects) {
      fieldDescription += `, 프로젝트: ${projects}`;
    }
    if (roles) {
      fieldDescription += `, 역할: ${roles}`;
    }

    const prompt = `
    당신은 채용 담당자입니다. 아래의 정보를 바탕으로 실무 중심의 면접 질문을 6개 생성해 주세요. 각 질문은 구체적이고 실제 면접에서 사용 가능한 수준으로 작성해 주세요.
    
    정보: 
    - 분야: ${category} - ${job}
    - 성과: ${achievements}
    - 경력: ${career}
    - 자격증: ${certificates}
    - 전공: ${major}
    - 프로젝트: ${projects}
    - 역할: ${roles}

    `;

    try {
      const openai = new OpenAI({
        apiKey: OPENAI_API_KEY.value(),
      });

      const chatCompletion = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
          { role: 'system', content: '당신은 채용 담당자입니다.' },
          { role: 'user', content: prompt },
        ],
        temperature: 0.7,
        max_tokens: 800,
      });

      const rawText = chatCompletion.choices[0].message.content ?? '';
      const questions = rawText
        .split('\n')
        .filter(line => line.match(/^\d+\./))
        .map(line => line.replace(/^\d+\.\s*/, '').trim());
      const finalQuestions = [
        ...staticQuestions,
        ...questions.slice(0, 6)
      ];
      await db.collection('interview_questions').doc(sessionId).set({
        field: fieldDescription,
        questions: finalQuestions,
        sessionId: sessionId, // sessionId 추가
        });
        

      console.log(` ${fieldDescription} 질문 저장 완료`);
    } catch (error) {
      console.error(' OpenAI 또는 Firestore 작업 중 오류:', error);
    }

    return null;
  });
