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
    const sessionId = data.sessionId || '';
    const questionCount = data.questionCount || 1; // 최소 1개 생성

    const category = data.category || '';
    const job = data.job || '';
    const achievements = data.achievements || '';
    const career = data.career || '';
    const certificates = data.certificates || '';
    const major = data.major || '';
    const projects = data.projects || '';
    const roles = data.roles || '';

    // 설명 필드 생성
    let fieldDescription = `${category} ${job}`;
    if (achievements) fieldDescription += `, 성과: ${achievements}`;
    if (certificates) fieldDescription += `, 자격증: ${certificates}`;
    if (career) fieldDescription += `, 경력: ${career}`;
    if (major) fieldDescription += `, 전공: ${major}`;
    if (projects) fieldDescription += `, 프로젝트: ${projects}`;
    if (roles) fieldDescription += `, 역할: ${roles}`;

    // ——— 정보 라인 동적 생성 ———
    const infoLines = [];
    // 반드시 분야·직무는 포함
    infoLines.push(`- 분야: ${category} - ${job}`);
    if (achievements) infoLines.push(`- 성과: ${achievements}`);
    if (career)       infoLines.push(`- 경력: ${career}`);
    if (certificates) infoLines.push(`- 자격증: ${certificates}`);
    if (major)        infoLines.push(`- 전공: ${major}`);
    if (projects)     infoLines.push(`- 프로젝트: ${projects}`);
    if (roles)        infoLines.push(`- 역할: ${roles}`);

    // ——— 프롬프트 작성 ———
    const prompt = `
        당신은 채용 담당자입니다.
        아래 정보만 참고해서 실무 중심의 면접 질문 ${questionCount}개를 생성하세요.
        정보가 없다면 자주 출제되는 면접질문을 출력해주세요.
        *절대 배경 설명 없이*, 반드시 번호("1.", "2.", ...)만 붙여 질문만 출력해 주세요.
        정보:
        ${infoLines.join('\n')}`;
    try {
      const openai = new OpenAI({ apiKey: OPENAI_API_KEY.value() });
      const chatCompletion = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
          { role: 'system', content: '당신은 채용 담당자입니다.' },
          { role: 'user', content: prompt }
        ],
        temperature: 0.7,
        max_tokens: 800,
      });

      const rawText = chatCompletion.choices[0].message.content || '';
      // 번호 매겨진 목록에서 질문 추출 후 제한
      // 1) 줄별로 trim() 후 빈 줄 제거
      const lines = rawText
        .split(/\r?\n/)
        .map(l => l.trim())
        .filter(l => l);

      // 2) 번호 매김된 줄이 있으면 그것만, 없으면 모든 줄을 사용
      let questions;
      if (lines.some(l => /^\d+\./.test(l))) {
        questions = lines
          .filter(l => /^\d+\./.test(l))
          .map(l => l.replace(/^\d+\.\s*/, ''));
      } else {
        questions = lines;
      }

      // 3) 최종 개수 제한
      questions = questions.slice(0, questionCount);
      // Firestore에 저장
      await db
        .collection('interview_questions')
        .doc(sessionId)
        .set({
          field: fieldDescription,
          questions: questions,
          sessionId: sessionId,
        });

      console.log(`${fieldDescription} - ${questionCount}개 질문 저장 완료`);
    } catch (error) {
      console.error('OpenAI 또는 Firestore 작업 중 오류:', error);
    }

    return null;
  });
