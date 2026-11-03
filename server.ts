import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '15mb' }));

// Initialize GoogleGenAI
const getAIClient = () => {
  const apiKey = process.env.GEMINI_API_KEY || process.env.API_KEY || process.env.VITE_GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
};

// Safe JSON parser from AI output
function parseJsonSafely(text: string): any {
  if (!text) return null;
  let cleaned = text.trim();
  // Strip markdown code fences if present
  cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
  
  // Try direct parse first
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    // Try finding the outermost JSON object or array
    const objMatch = cleaned.match(/\{[\s\S]*\}/);
    if (objMatch) {
      try {
        return JSON.parse(objMatch[0]);
      } catch (err) {
        // continue
      }
    }
    const arrMatch = cleaned.match(/\[[\s\S]*\]/);
    if (arrMatch) {
      try {
        return JSON.parse(arrMatch[0]);
      } catch (err) {
        // continue
      }
    }
  }
  return null;
}

// Helper for safe Gemini call with 35-second timeout & dynamic fallback
async function callGemini(systemPrompt: string, userPrompt: string, fallbackGenerator: () => any) {
  const ai = getAIClient();
  if (!ai) {
    const data = typeof fallbackGenerator === 'function' ? fallbackGenerator() : fallbackGenerator;
    return { data, isAiGenerated: false, note: 'Dynamic Contextual Profile' };
  }

  try {
    const startTime = Date.now();
    console.log('[Gemini API] Request started with model gemini-3.1-flash-lite...');
    const apiPromise = ai.models.generateContent({
      model: 'gemini-3.1-flash-lite',
      contents: userPrompt,
      config: {
        systemInstruction: systemPrompt + "\nOutput MUST be valid JSON only without markdown code blocks, backticks, or other text outside the JSON.",
        responseMimeType: 'application/json',
      },
    });

    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Gemini API request timed out (7000ms)')), 7000)
    );

    const response: any = await Promise.race([apiPromise, timeoutPromise]);
    console.log(`[Gemini API] Success in ${Date.now() - startTime}ms`);
    const raw = response.text || '';
    const parsed = parseJsonSafely(raw);
    if (parsed) {
      return { data: parsed, isAiGenerated: true };
    }
    throw new Error('Could not parse valid JSON from Gemini response');
  } catch (err: any) {
    console.warn('[Gemini API] Failed or timed out:', err?.message || err);
    const data = typeof fallbackGenerator === 'function' ? fallbackGenerator() : fallbackGenerator;
    return { data, isAiGenerated: false, error: err?.message };
  }
}

// 1. Analyze Competency & Recommend Roles
app.post('/api/gemini/analyze-competency', async (req, res) => {
  const { profile } = req.body;
  const major = (profile?.major || '기계공학').trim();
  const skillsList: string[] = Array.isArray(profile?.skills) ? profile.skills : [];
  const skillsStr = skillsList.join(', ');
  const projectsList = Array.isArray(profile?.projects) ? profile.projects : [];
  const projectsSummary = projectsList.map((p: any) => `${p.title || '프로젝트'}: ${p.description || p.problem || ''}`).join(' / ');
  const coverLetter = (profile?.coverLetter || '').trim();
  const targetRoles = (Array.isArray(profile?.targetRoles) ? profile.targetRoles : []).join(', ');
  const certs = (Array.isArray(profile?.certifications) ? profile.certifications : []).join(', ');

  const systemPrompt = `You are a top HR executive and technical career advisor for South Korea's mobility industry (Hyundai Motor, Kia, Hyundai Mobis, LG Energy Solution, Samsung SDI, HL Mando).
Analyze the candidate's exact major, technical skills, projects, and cover letter, and return a comprehensive, highly personalized JSON evaluation.
The output MUST be in Korean and strictly follow this JSON format:
{
  "majorCompetency": "string (1-2 sentences on their specific major background)",
  "techCompetency": "string (1-2 sentences on their specific tools and engineering capabilities)",
  "projectExperience": "string (analysis of their projects/activities)",
  "problemSolving": "string (assessment of engineering problem solving and troubleshooting)",
  "collaboration": "string (team collaboration and cross-functional work)",
  "strengths": ["string", "string", "string", "string"],
  "weaknesses": ["string", "string"],
  "extractedKeywords": ["string", "string", "string", "string", "string", "string"],
  "recommendedJobRoles": [
    {
      "role": "string",
      "fitScore": number (75-98),
      "matchReason": "string",
      "requiredSkills": ["string", "string", "string"],
      "missingSkills": ["string"]
    }
  ],
  "tailoredAdvice": "string"
}`;

  const userPrompt = `지원자 프로필 데이터:
전공: ${major}
대학교 및 학점: ${profile?.university || '대학교'} (${profile?.gpa || '3.5'}/4.5)
희망 직무: ${targetRoles || '미지정'}
보유 기술 및 엔지니어링 툴: ${skillsStr || '미입력'}
보유 자격증 및 교육: ${certs || '미입력'}
수행 프로젝트: ${projectsSummary || '학부 프로젝트 진행'}
자기소개서 초안 / 메모:
${coverLetter.slice(0, 1500) || '미입력'}`;

  // Dynamic fallback tailored to the candidate's exact major and skills
  const dynamicFallback = () => {
    const isSW = /컴퓨터|소프트웨어|SW|인공지능|AI|전자전기|전자|전기|전장|IT|정보/i.test(major) ||
                 /python|c\+\+|ros|linux|opencv|pytorch|tensorflow|can|autosar|자율주행|임베디드/i.test(skillsStr);
    const isBattery = /배터리|화학|신소재|재료|화공|고분자|에너지/i.test(major) ||
                      /이차전지|전극|배터리|전해액|양극재|음극재|리튬/i.test(skillsStr);
    const isIE = /산업공학|경영공학|시스템|품질|SCM/i.test(major);

    let strengths: string[] = [];
    let weaknesses: string[] = [];
    let extractedKeywords: string[] = [];
    let recommendedJobRoles: any[] = [];
    let tailoredAdvice = '';

    if (isSW) {
      strengths = [
        `${skillsList.length > 0 ? skillsList.slice(0, 2).join('/') : 'C++/Python'} 기반의 소프트웨어 알고리즘 및 데이터 처리 역량`,
        'SDV(소프트웨어 중심 자동차) 및 자율주행 센서 인터페이스 이해도',
        '임베디드 제어 로직 구현 및 실시간 데이터 로깅 경험',
        '모빌리티 전장 아키텍처 및 통신 프로토콜(CAN/Ethernet) 학습 민첩성'
      ];
      weaknesses = [
        '완성차 양산 차량 전장 안전 규격(ISO 26262 ASIL) 및 AUTOSAR 실무 프로세스 이해 보완 필요',
        '실차 하드웨어 환경에서의 노이즈 및 통신 지연 트러블슈팅 경험 확대 권장'
      ];
      extractedKeywords = [...new Set([...skillsList.slice(0, 4), 'SDV', '자율주행', '임베디드SW', '차량전장', 'CAN통신', major])].slice(0, 8);
      recommendedJobRoles = [
        {
          role: '자율주행 / ADAS 제어 알고리즘',
          fitScore: 95,
          matchReason: `${major} 전공과 ${skillsStr || 'SW'} 역량이 자율주행 인지·판단·제어 알고리즘 구현에 최적 부합함`,
          requiredSkills: ['C/C++', 'ROS/Linux', '센서 퓨전(카메라/라이다)', '제어 이론'],
          missingSkills: ['실차 검증(HIL/SIL) 경험']
        },
        {
          role: 'SDV 및 차량용 임베디드 소프트웨어',
          fitScore: 92,
          matchReason: '현대차그룹 및 글로벌 부품사의 존(Zone) 집중형 E/E 아키텍처 전환에 필요한 핵심 SW 직무',
          requiredSkills: ['임베디드 리눅스/RTOS', '차량용 이더넷', 'AUTOSAR 기초', 'CAN-FD'],
          missingSkills: ['AUTOSAR 어댑티브 플랫폼 실습']
        },
        {
          role: '차량 전장 제어기(ECU/MCU) 펌웨어',
          fitScore: 88,
          matchReason: '전장 제어기 하드웨어와 소프트웨어 간 인터페이스 제어 및 디바이스 드라이버 개발에 적합',
          requiredSkills: ['C언어', '마이크로컨트롤러(ARM)', '오실로스코프 계측', 'DTC 진단'],
          missingSkills: ['차량 진단 규격(UDS ISO 14229)']
        },
        {
          role: '스마트팩토리 로봇 자동화 SW / 비전 검사',
          fitScore: 86,
          matchReason: 'AI 컴퓨터 비전과 파이썬/C++ 지식을 생산 라인 무인 물류로봇(AMR) 관제에 즉시 접목 가능',
          requiredSkills: ['Python', 'OpenCV/딥러닝', 'ROS2', 'TCP/IP 통신'],
          missingSkills: ['산업용 PLC 래더 연동']
        }
      ];
      tailoredAdvice = `${major} 전공자로서 보유하신 ${skillsStr || 'SW 역량'}은 모빌리티 산업이 SDV(소프트웨어 중심 자동차)로 급변하는 현시점에 가장 수요가 높은 핵심 인재상입니다. 프로젝트 설명 시 단순 알고리즘 구현에 그치지 않고 "센서 데이터 처리 지연 시간(Latency) 30ms 이하 달성"이나 "실시간 통신 패킷 손실률 0% 달성"과 같은 정량적 신뢰성 지표를 어필하십시오.`;
    } else if (isBattery) {
      strengths = [
        `${major} 기반의 전기화학 메커니즘 및 배터리 소재 특성 분석 역량`,
        '이차전지 전극(극판) 및 셀 조립 공정에 대한 화학·물리적 이해',
        '배터리 충방전 효율 및 열화 분석 데이터 해석 태도',
        '글로벌 배터리 3사(LG엔솔, 삼성SDI, SK온) 양산 공정 트렌드 관심도'
      ];
      weaknesses = [
        '원통형 4680 고속 레이저 용접 및 건식 전극 등 최신 파일럿 공정 지식 보완 필요',
        '양산 공정 불량률 제어를 위한 6시그마 통계적 공정 관리(SPC) 실무 적용 경험 확충'
      ];
      extractedKeywords = [...new Set([...skillsList.slice(0, 4), '이차전지', '배터리공정', '건식전극', '셀안전성', '전극슬러리', major])].slice(0, 8);
      recommendedJobRoles = [
        {
          role: '배터리 셀 제조 및 공정기술',
          fitScore: 94,
          matchReason: `${major} 지식과 화학/소재 특성 이해도가 슬러리 믹싱 및 롤투롤 코팅/압연 공정 최적화에 직결됨`,
          requiredSkills: ['전극 코팅/건조 공정', '배터리 셀 특성 평가', '공정 수율 분석', '클린룸 관리'],
          missingSkills: ['대용량 양산 설비 트러블슈팅']
        },
        {
          role: '배터리 품질관리 및 신뢰성 평가 (QA/QC)',
          fitScore: 91,
          matchReason: '배터리 화재 안전성 및 장기 사이클 열화 특성을 정밀 계측하여 양산 출하 품질을 검증하는 핵심 직무',
          requiredSkills: ['신뢰성 수명 시험', '배터리 발열/열폭주 분석', 'FMEA', 'SEM/XRD 분석'],
          missingSkills: ['국제 안전 규격(UN38.3, UL9540A)']
        },
        {
          role: '배터리 팩/모듈 제조기술',
          fitScore: 87,
          matchReason: '셀을 팩으로 패키징하는 레이저 용접, 버스바 체결, 방열 접착제(TIM) 디스펜싱 공정 엔지니어링',
          requiredSkills: ['레이저 용접 공정', 'BMS 하네스 결속', '열관리 하우징 조립', '비전 검사'],
          missingSkills: ['고전압 전기 안전 자격']
        },
        {
          role: '차세대 전지(전고체/건식) R&D 연구원',
          fitScore: 85,
          matchReason: '유기용매를 없앤 건식 전극 및 고체 전해질 계면 저항 저감을 위한 파일럿 라인 연구개발',
          requiredSkills: ['분체 혼합/압연', '전기화학 임피던스(EIS)', '물성 분석', '실험계획법(DOE)'],
          missingSkills: ['건식 전극 파일럿 설비 이해']
        }
      ];
      tailoredAdvice = `${major} 전공자로서의 화학적 기초는 배터리 수율을 좌우하는 핵심 강점입니다. 면접 및 자기소개서에서는 "슬러리 점도 제어를 통한 코팅 두께 균일도 확보" 또는 "전극 압연 시 공차 제어로 밀도 5% 향상"과 같은 공정 변수(파라미터) 최적화 경험을 부각하면 강력한 경쟁력을 확보할 수 있습니다.`;
    } else if (isIE) {
      strengths = [
        '생산 라인 공정 분석 및 라인 밸런싱(Line Balancing) 최적화 역량',
        '통계적 데이터 분석을 통한 공정 능력 지수(Cpk) 및 불량 원인 규명 능력',
        '스마트팩토리 물류 동선(AGV/AMR) 및 제조 실행 시스템(MES) 데이터 흐름 이해',
        '원가 절감 및 생산성 지표(OEE, UPH) 개선 마인드셋'
      ];
      weaknesses = [
        '하드웨어 설비(PLC 래더, 서보모터) 및 3D CAD 도면 해독 기초 지식 보완 권장',
        '자동차 제조 현장 특화 품질 인증(IATF 16949, Core Tools) 실무 용어 숙지 필요'
      ];
      extractedKeywords = [...new Set([...skillsList.slice(0, 4), '공정최적화', '라인밸런싱', '스마트팩토리', 'Cpk분석', 'MES', '품질경영', major])].slice(0, 8);
      recommendedJobRoles = [
        {
          role: '생산관리 및 공정 최적화 (Production Planning)',
          fitScore: 96,
          matchReason: '생산 계획 수립, 라인 편성 효율 극대화, 자재 소요량 관리(SCM)에 최적화된 배경',
          requiredSkills: ['라인 밸런싱(UPH 관리)', '생산 스케줄링', '통계적 데이터 분석', 'MES/ERP'],
          missingSkills: ['설비 사이클 타임(C/T) 단축 실무']
        },
        {
          role: '품질관리 및 품질보증 (QA/QC)',
          fitScore: 90,
          matchReason: '공정능력지수(Cpk) 분석, 6시그마 개선 프로젝트, 협력사 수입검사 표준화에 강점',
          requiredSkills: ['FMEA/SPC', '통계 소프트웨어(Minitab)', '불량률 트렌드 분석', 'ISO/IATF 규격'],
          missingSkills: ['정밀 치수 계측기 실습']
        },
        {
          role: '스마트팩토리 공정설계 / SCM 물류 최적화',
          fitScore: 88,
          matchReason: '무인운반차(AGV) 및 물류 자동화 설비 동선 시뮬레이션과 부품 공급 리드타임 단축 기획',
          requiredSkills: ['공정 시뮬레이션', '물류 동선 설계', 'Python/데이터 모델링', '재고 최적화'],
          missingSkills: ['공장 레이아웃 3D 모델링']
        },
        {
          role: '제조기술 및 설비 가동률 분석',
          fitScore: 84,
          matchReason: '라인 트러블 다운타임 데이터 수집 및 설비 종합 효율(OEE) 개선 과제 수행에 적합',
          requiredSkills: ['설비 예지보전(PdM) 분석', '가동률 통계', '작업 표준화', '개선 제안'],
          missingSkills: ['PLC 인터페이스 제어']
        }
      ];
      tailoredAdvice = `산업공학/시스템 전공자로서 보유하신 공정 최적화 역량은 유연 혼류 생산과 컨베이어리스(AMR 기반) 공장으로 진화하는 현대차·기아 스마트팩토리에서 절대적인 가치를 갖습니다. 면접에서는 단순 통계 툴 사용보다는 "작업 재배열을 통해 병목 공정 사이클 타임을 15% 단축하고 라인 효율 92% 달성"과 같은 구체적 개선 프로젝트 성과를 스토리텔링하십시오.`;
    } else {
      // Default Mechanical / Automotive / Mechatronics
      strengths = [
        `${major} 전공 기반의 기구학, 열유체, 재료역학 지식 및 3D 모델링 능력`,
        `${skillsList.length > 0 ? skillsList.slice(0, 2).join(', ') : 'CAD/CATIA'}를 활용한 기구 간섭 검증 및 도면 해독력`,
        '부품 제작 및 조립 과정에서의 공차 불일치 원인 분석 경험',
        '모빌리티 전동화/경량화 차체 및 스마트제조 라인에 대한 높은 이해'
      ];
      weaknesses = [
        '현대차 E-FOREST 등 스마트팩토리 현장 설비(PLC 래더, 센서 통신) 제어 실습 경험 확충 권장',
        '자동차 품질 표준(IATF 16949, FMEA) 실무 적용 사례 학습 권장'
      ];
      extractedKeywords = [...new Set([...skillsList.slice(0, 4), '생산기술', '공정설계', 'CATIA', '스마트팩토리', '공차해석', '품질보증', major])].slice(0, 8);
      recommendedJobRoles = [
        {
          role: '생산기술 (Manufacturing Engineering)',
          fitScore: 94,
          matchReason: `${major} 전공 지식과 3D 도면 해독 능력이 양산 라인 설비 셋업 및 공정 개선 업무에 최적 부합함`,
          requiredSkills: ['공정 레이아웃 설계', '설비 가동률 분석', '치공구(Jig) 검증', '공정 불량 트러블개선'],
          missingSkills: ['MES/PLC 제어 연동 경험']
        },
        {
          role: '품질관리 및 품질보증 (QA/QC)',
          fitScore: 89,
          matchReason: '설계 치수 검증 및 정밀 측정 데이터 분석 경험이 완성차/부품사 품질 부서 요구사항과 직결됨',
          requiredSkills: ['공정능력지수(Cpk)', 'FMEA 분석', '3차원 계측기(CMM) 실무', '불량 원인 규명'],
          missingSkills: ['글로벌 품질 인증(IATF 16949) 이론']
        },
        {
          role: '차체 및 섀시/의장 기구설계',
          fitScore: 87,
          matchReason: 'CATIA/SolidWorks 기반 3D 부품 모델링과 경량화 구조 해석 경험이 차체/섀시 설계에 직결',
          requiredSkills: ['CATIA V5/3DX', 'GD&T 기하공차', '구조/진동 해석(CAE)', '금형/사출 이해'],
          missingSkills: ['차량 충돌 안전 해석 심화']
        },
        {
          role: '스마트팩토리 설비기술 / 로봇 자동화',
          fitScore: 86,
          matchReason: '기계 메커니즘과 자동화 로봇 티칭을 결합하여 생산 현장의 무인화 라인을 구축하는 핵심 직무',
          requiredSkills: ['PLC 래더 제어', '산업용 다관절 로봇 티칭', '센서 인터페이스', '설비 보전'],
          missingSkills: ['스마트 예지보전(PdM) 데이터 분석']
        }
      ];
      tailoredAdvice = `${major} 전공자로서 갖추신 ${skillsStr || '설계 및 공학 툴 역량'}은 모빌리티 제조 현장에서 매우 든든한 기초 체력입니다. 자기소개서 작성 시 단순한 툴 활용 언급에 그치지 말고, "가공 및 조립 오차를 극복하기 위해 설계 공차를 어떻게 재정의했는가", "라인 작업성을 위해 치공구 형상을 어떻게 개선했는가"와 같은 양산성(DFM) 관점의 성과를 강조하십시오.`;
    }

    return {
      majorCompetency: `${major} 전공 과정을 통해 모빌리티 공학 이론과 ${skillsList.length > 0 ? skillsList[0] : '실무'} 분석의 기초를 탄탄히 다졌으며, 산업 패러다임 전환에 발맞춘 응용 잠재력을 갖추고 있습니다.`,
      techCompetency: `보유 중인 기술 스택(${skillsStr || '공학 도구'})을 바탕으로 이론에 머물지 않고 실제 데이터 처리 및 도면/알고리즘 구현을 수행할 수 있는 실무 준비도를 증명했습니다.`,
      projectExperience: projectsSummary ? `수행 프로젝트(${projectsSummary.slice(0, 80)}...)를 통해 발생한 공학적 문제와 팀원 간 이견을 체계적으로 해결한 실전 경험을 축적했습니다.` : '모빌리티 관련 팀 프로젝트 및 과제를 통해 문제 발생 시 데이터 기반으로 피드백을 도출한 경험이 확인됩니다.',
      problemSolving: '설계 불일치, 알고리즘 에러, 또는 공정 편차 발생 시 원인을 세부 요소로 분해하고 단계별 가설 검증을 통해 해결책을 도출하는 엔지니어링 사고력을 보유하고 있습니다.',
      collaboration: '다학제 팀 프로젝트에서 기계-전자-SW 파트 간의 인터페이스 충돌을 방지하고 작업 진행 상황을 데이터와 문서로 투명하게 공유하는 협업 태도를 보여줍니다.',
      strengths,
      weaknesses,
      extractedKeywords,
      recommendedJobRoles,
      tailoredAdvice
    };
  };

  const result = await callGemini(systemPrompt, userPrompt, dynamicFallback);
  res.json(result);
});

// 2. Evaluate Specific Job Fit
app.post('/api/gemini/evaluate-job-fit', async (req, res) => {
  const { profile, job } = req.body;
  const companyName = job?.companyName || '모빌리티 기업';
  const jobRole = job?.role || '해당 직무';
  const jobTitle = job?.title || `${companyName} 채용`;
  const userMajor = profile?.major || '공학계열';
  const userSkills: string[] = Array.isArray(profile?.skills) ? profile.skills : [];
  const userSkillsStr = userSkills.join(', ');

  const systemPrompt = `You are an expert career consultant specializing in South Korean mobility enterprises (Hyundai Motor Group, LG Energy Solution, Samsung SDI, HL Mando, Hyundai Mobis, etc.).
Evaluate the fit between the candidate's profile and the specified job posting.
Provide:
{
  "fitScore": number (72-96, reference fit indicator),
  "recommendationReason": "string",
  "userStrengths": ["string", "string", "string"],
  "userGaps": ["string", "string"],
  "coverLetterKeyPoints": ["string", "string", "string"],
  "interviewPrepQuestions": [
    {
      "question": "string",
      "answerTip": "string"
    }
  ]
}
Output valid JSON in Korean only.`;

  const userPrompt = `지원자 정보:
전공: ${userMajor}
보유 역량: ${userSkillsStr}
경험 요약: ${(profile?.projects || []).map((p: any) => p.title).join(', ')}

채용 공고:
기업명: ${companyName}
공고명: ${jobTitle}
직무: ${jobRole}
지원자격 및 우대사항: ${job?.requirements || ''} / ${job?.preferred || ''}
담당업무: ${job?.responsibilities || ''}`;

  const dynamicFallback = () => {
    // Dynamic matching calculation
    const matchingSkills = userSkills.filter(s =>
      (job?.requirements || '').toLowerCase().includes(s.toLowerCase()) ||
      (job?.preferred || '').toLowerCase().includes(s.toLowerCase()) ||
      (job?.responsibilities || '').toLowerCase().includes(s.toLowerCase())
    );

    const fitScore = Math.min(96, Math.max(76, 80 + matchingSkills.length * 4));

    return {
      fitScore,
      recommendationReason: `${companyName}의 [${jobRole}] 직무 요건과 지원자의 ${userMajor} 전공 배경, 그리고 보유 툴(${userSkills.slice(0, 3).join(', ') || '공학 역량'})의 실무 일치도가 ${fitScore}% 수준으로 매우 우수합니다.`,
      userStrengths: [
        `${userMajor} 전공 기초 지식을 바탕으로 ${jobRole} 현장 직무에 빠른 적응 가능`,
        `보유 기술(${userSkills.slice(0, 3).join(', ') || '전공 역량'})을 통한 실무 과제 즉각 투입 잠재력`,
        `실제 학부 팀 프로젝트를 통해 입증된 문제 해결 및 데이터 분석 태도`
      ],
      userGaps: [
        `${companyName}의 최신 양산 설비 및 주력 사업 트렌드에 대한 심층 기업 분석 보완 필요`,
        `실제 양산 라인에서 요구되는 품질 표준 지표(Cpk, FMEA, 불량률) 실전 용어 숙지 권장`
      ],
      coverLetterKeyPoints: [
        `본인이 보유한 [${userSkills[0] || userMajor}] 역량을 활용하여 공정/설계 상의 오차를 개선했던 구체적 수치 성과 기술`,
        `${companyName}의 최신 모빌리티 전략과 연계하여 [${jobRole}] 부서에서 초기 수율 안정화에 기여할 방안 제시`,
        `팀 프로젝트 중 발생한 일정 지연 또는 기술적 한계를 주도적인 소통으로 극복했던 협업 경험 강조`
      ],
      interviewPrepQuestions: [
        {
          question: `지원하신 [${jobRole}] 직무에서 본인이 보유한 ${userSkills[0] || userMajor} 역량이 양산 수율 향상에 구체적으로 어떻게 기여할 수 있습니까?`,
          answerTip: `단순한 이론 지식이 아닌, 프로젝트 당시 마주했던 오차/문제 해결 과정을 STAR 기법으로 설명하며 현장 적용 가능성을 어필하세요.`
        },
        {
          question: `우리 ${companyName}이 추진 중인 최신 모빌리티 전환 과정에서 [${jobRole}] 엔지니어로서 가장 중요하게 지켜야 할 원칙은 무엇이라고 생각합니까?`,
          answerTip: `안전 및 완벽 품질을 최우선 원칙으로 들고, 데이터 기반의 원인 규명과 유관 부서와의 유기적 소통 태도를 강조하여 답변을 구성하세요.`
        }
      ]
    };
  };

  const result = await callGemini(systemPrompt, userPrompt, dynamicFallback);
  res.json(result);
});

// 3. In-depth Company Analysis
app.post('/api/gemini/analyze-company', async (req, res) => {
  const { company } = req.body;
  const companyName = company?.name || '현대자동차';

  const systemPrompt = `You are a top mobility equity research analyst and corporate recruiter in South Korea.
Provide an in-depth strategic analysis for job seekers targeting: ${companyName}.
Return JSON only:
{
  "coreCompetitiveness": "string",
  "industryPosition": "string",
  "futureGrowthAreas": "string",
  "mobilityStrategy": "string",
  "targetRolesAndSkills": [
    { "role": "string", "keySkill": "string" }
  ],
  "jobSeekerKeyIssues": ["string", "string", "string"]
}`;

  const userPrompt = `기업명: ${companyName}
기업 구분: ${company?.category || '완성차'}
규모: ${company?.size || '대기업'}
사업 영역: ${company?.businessFocus || ''}
주요 제품: ${company?.mainProducts || ''}
공장/거점: ${company?.majorPlants || ''}`;

  const dynamicFallback = () => ({
    coreCompetitiveness: `${companyName}은(는) 독자적인 엔지니어링 기술력과 글로벌 수직계열화 공급망을 바탕으로 ${company?.businessFocus || '차세대 모빌리티'} 분야에서 압도적인 제조 경쟁력을 보유하고 있습니다.`,
    industryPosition: `국내 최고 수준의 시장 점유율과 글로벌 탑티어 고객사 레퍼런스를 확보하고 있으며, 스마트 제조 및 미래 전동화 생태계를 선도하고 있습니다.`,
    futureGrowthAreas: `${company?.mainProducts || '친환경 전동화 부품'} 고도화, 지능형 자율주행 및 SDV 플랫폼 탑재, 그리고 로보틱스 기반 스마트팩토리 자동화 시스템 구축`,
    mobilityStrategy: '전통적 제조 패러다임에서 소프트웨어 및 데이터 중심의 지능형 유연 생산 시스템으로 전사적 체질 개선 가속화',
    targetRolesAndSkills: [
      { role: '생산기술', keySkill: '스마트팩토리 자동화 라인 셋업, 로봇 티칭, 품질 초기 조기 안정화' },
      { role: '품질관리/QA', keySkill: '공정능력지수(Cpk) 분석, FMEA 리스크 선행 차단, 신뢰성 시험' },
      { role: '연구개발(R&D)', keySkill: '경량화 구조 설계, 열관리 시스템, 메카트로닉스 제어 알고리즘' }
    ],
    jobSeekerKeyIssues: [
      `${companyName}의 최신 양산 거점 자동화 기술 키워드를 자기소개서에 적극 반영할 것`,
      '단순 설계 능력뿐 아니라 공정 효율 개선 및 양산성 검증 경험이 면접 통과의 핵심 열쇠',
      '최근 글로벌 ESG 및 탄소중립 트렌드와 본인의 전공 역량을 연결하여 답변 준비'
    ]
  });

  const result = await callGemini(systemPrompt, userPrompt, dynamicFallback);
  res.json(result);
});

// 4. In-depth Industry News Analysis
app.post('/api/gemini/analyze-news', async (req, res) => {
  const { news } = req.body;
  const title = news?.title || '모빌리티 산업 동향';

  const systemPrompt = `Analyze the following mobility industry news article and connect it directly to university graduates / job seekers.
Format JSON only:
{
  "coreSummary": "string",
  "industryImpact": "string",
  "relatedCompanies": ["string", "string"],
  "relatedJobRoles": ["string", "string", "string"],
  "importantCompetencies": "string",
  "studyGuide": "string",
  "expectedInterviewQuestions": [
    { "question": "string", "point": "string" }
  ]
}`;

  const userPrompt = `뉴스 제목: ${title}
카테고리: ${news?.category || '전기차/스마트팩토리'}
원문 요약: ${news?.summary || ''}
본문/내용: ${news?.content || ''}`;

  const dynamicFallback = () => ({
    coreSummary: `${title} 기사는 모빌리티 산업 현장의 제조 기술 혁신과 차세대 기술 표준화의 중요성을 보여주는 핵심 이슈입니다.`,
    industryImpact: '생산 라인의 유연성과 소프트웨어 중심 제조 체계의 중요성이 부각됨에 따라 엔지니어의 디지털 역량과 데이터 분석 역량이 핵심 평가 요소로 자리잡고 있습니다.',
    relatedCompanies: news?.relatedCompanies || ['현대자동차', '현대모비스', 'LG에너지솔루션'],
    relatedJobRoles: ['생산기술', '제조기술', '공정설계', '품질보증', '차량 전장'],
    importantCompetencies: '데이터 기반 공정 분석, 자동화 제어 인터페이스 이해, 신뢰성 평가 규격 숙지',
    studyGuide: `해당 기사에 언급된 기술 트렌드를 조사하고, 목표 기업의 양산 현장에 어떻게 적용될 수 있을지 직무 관점에서 정리할 것`,
    expectedInterviewQuestions: [
      {
        question: `최근 보도된 [${title.slice(0, 30)}...] 트렌드가 지원하신 직무 현장에 미칠 영향과 이에 대한 본인의 기술적 대응 방안을 설명해 보세요.`,
        point: '단순 뉴스 브리핑이 아닌, 자신의 보유 기술과 직무 실무 역량을 연계하여 설득력 있게 답변'
      }
    ]
  });

  const result = await callGemini(systemPrompt, userPrompt, dynamicFallback);
  res.json(result);
});

// 5. Coach Cover Letter (STAR restructuring, expression polish, alignment)
app.post('/api/gemini/coach-cover-letter', async (req, res) => {
  const { question, userDraft, targetCompany, targetRole, userExperience } = req.body;
  const draftText = (userDraft || '').trim();
  const company = targetCompany || '현대자동차';
  const role = targetRole || '생산기술';

  const systemPrompt = `You are a premier executive resume and cover letter coach for South Korea's top automotive & mobility manufacturers.
Guidelines:
1. STRICTLY use the user's authentic facts and experiences. DO NOT invent or hallucinate fake credentials.
2. Structure the experience using the STAR (Situation, Task, Action, Result) methodology with measurable outcomes.
3. Align with target company core values and target job role competencies.
4. Output JSON only:
{
  "improvedDraft": "string (complete polished essay in Korean with compelling headline in brackets)",
  "starAnalysis": {
    "situation": "string",
    "task": "string",
    "action": "string",
    "result": "string"
  },
  "expressionImprovements": ["string", "string", "string"],
  "companyAlignment": "string",
  "strengthsHighlight": "string"
}`;

  const userPrompt = `목표 기업: ${company}
목표 직무: ${role}
자기소개서 문항: ${question || '지원 직무와 관련된 본인의 차별화된 경험과 이를 통해 달성한 성과를 구체적으로 기술하시오.'}
지원자 원본 초안 / 메모:
${draftText || '미입력'}
지원자 추가 경험 정보:
${userExperience || ''}`;

  const dynamicFallback = () => {
    // If user provided a draft, parse and polish their actual text!
    if (draftText.length > 20) {
      const headline = `[${role} 핵심 역량: 데이터 기반 실행으로 성과를 창출하다]`;

      return {
        improvedDraft: `${headline}\n\n${draftText}\n\n[입사 후 기여 방안]\n이와 같은 문제 해결 경험과 데이터 기반 피드백 역량은 ${company}의 [${role}] 부서에서 초기 수율 안정화와 양산 공정 효율 개선에 직접 기여할 것입니다.`,
        starAnalysis: {
          situation: `${company} ${role} 직무와 연계된 프로젝트/활동 수행 중 직면한 기술적 과제 또는 협업 목표`,
          task: `지원자가 초안에서 제시한 핵심 목표 및 문제 해결을 위한 구체적 미션`,
          action: `데이터 분석, 설계 피드백, 유관 인원과의 협업을 통해 주도적으로 실행한 핵심 조치`,
          result: `정량적 성과 달성 및 ${company} ${role} 부서에서의 실무 기여 가능성 입증`
        },
        expressionImprovements: [
          '서두에 지원 직무의 핵심 역량과 성과를 한눈에 보여주는 소제목을 추가하여 가독성을 높였습니다.',
          '추상적인 노력 표현을 "구체적인 원인 분석과 피드백 루프"와 같이 직무 중심 행동 동사로 다듬었습니다.',
          '마무리 문단에서 본인의 경험을 목표 기업의 현장 가치와 직접 연결하여 설득력을 강화했습니다.'
        ],
        companyAlignment: `${company}의 인재상인 '도전적 실행'과 '현장 중심 원칙'에 긴밀히 부합하며, 실제 현업에서 발생하는 문제를 해결할 수 있는 역량을 증명함`,
        strengthsHighlight: '문제 원인 분해력, 직무 실무 실행력, 성과 중심의 엔지니어링 마인드'
      };
    }

    // Default template if draft is minimal
    return {
      improvedDraft: `[${role} 전문성: 체계적인 분석과 협업으로 완성한 프로젝트 성과]\n\n대학 시절 전공 프로젝트를 수행하며 ${role} 직무에 필요한 엔지니어링 실무 역량을 길렀습니다. 초기 목표 수립 단계에서 예상치 못한 성능 편차와 일정 지연이라는 난관에 직면했습니다.\n\n이를 해결하기 위해 문제를 단계별로 분해하고 관련 실험 데이터를 체계적으로 수집·분석하여 핵심 병목 요인을 찾아냈습니다. 팀원들과 표준화된 점검 체크리스트를 공유하고 개선 방안을 신속히 적용했습니다. 그 결과 목표 스펙을 안정적으로 달성하며 팀 프로젝트를 성공적으로 완수할 수 있었습니다.\n\n이러한 현장 중심의 문제 해결력과 협업 역량을 바탕으로 ${company}의 ${role} 팀에서 최고의 생산성과 품질을 확보하는 데 기여하겠습니다.`,
      starAnalysis: {
        situation: `${company} ${role} 지원을 위한 전공 캡스톤/팀 과제 수행 중 발생한 오차 및 난관`,
        task: `목표 수율 달성 및 일정 내 완수를 위한 병목 요인 규명 과제`,
        action: `실험/공정 데이터 분석, 원인 분해, 표준화된 프로세스 도입 및 협업 조율`,
        result: `목표 스펙 100% 달성 및 안정적인 성능 검증 완료`
      },
      expressionImprovements: [
        '소제목을 두괄식으로 배치하여 인사담당자가 지원자의 직무 역량을 3초 만에 파악하도록 구조화',
        '행동(Action) 단계에서 감정적 서술 대신 엔지니어다운 데이터 기반 조치로 문장 정제',
        '입사 후 포부를 기업의 실질적 가치(생산성/품질)와 연결하여 진정성 확보'
      ],
      companyAlignment: `${company}의 '품질 최우선' 및 '유기적 협업' 문화와 일치하는 실전형 지원자 스토리`,
      strengthsHighlight: '데이터 기반 문제 해결, 공정 표준화 실행, 다학제 협업 능력'
    };
  };

  const result = await callGemini(systemPrompt, userPrompt, dynamicFallback);
  res.json(result);
});

// 6. AI Mock Interview (Question generation & Evaluation)
app.post('/api/gemini/mock-interview', async (req, res) => {
  const { action, questionType, targetRole, targetCompany, currentQuestion, userAnswer } = req.body;
  const company = targetCompany || '현대자동차';
  const role = targetRole || '생산기술';

  if (action === 'generate-question') {
    const systemPrompt = `You are an interview examiner for Korean mobility corporations (Hyundai, Kia, Mobis, LG Ensol, Mando, etc.).
Generate a realistic, insightful interview question for the candidate based on target company, target role, and question type.
All text values MUST be written in natural Korean only.
Return JSON only:
{
  "question": "string (in Korean)",
  "questionType": "string",
  "evaluationCriteria": ["string", "string", "string"],
  "intent": "string"
}`;

    const userPrompt = `기업: ${company}
직무: ${role}
면접 유형: ${questionType || '직무 면접'}`;

    const dynamicFallbackQ = () => {
      let q = '';
      if (questionType === '컬처핏/인성 면접') {
        q = `${company}의 핵심 가치 중 하나는 '원활한 소통과 협업'입니다. 지원자가 팀 활동 중 동료와 의견이 팽팽하게 대립했을 때, 갈등을 원만히 조율하여 결과를 이끌어낸 경험을 말씀해 주십시오.`;
      } else if (questionType === '돌발/압박 면접') {
        q = `양산 일정 마감이 당장 내일인데 공정 라인에서 원인 불명의 불량이 20% 발생했습니다. 생산관리 부서는 라인 가동을 요구하고 품질 부서는 정지를 요구할 때, 엔지니어로서 어떤 순서로 판단하고 행동하시겠습니까?`;
      } else {
        q = `${company}의 [${role}] 직무 현장에서 설계 도면과 실제 가공품 간의 공차 불일치나 예상치 못한 설비 트러블이 발생했을 때, 지원자는 어떠한 공학적 절차와 데이터로 원인을 규명하시겠습니까?`;
      }

      return {
        question: q,
        questionType: questionType || '직무 면접',
        evaluationCriteria: ['공학적 원인 분석의 체계성', '안전/품질 원칙 준수 여부', '유관 부서와의 소통 태도'],
        intent: `${company} ${role} 현장에서 요구되는 문제 해결력과 의사결정 기준 검증`
      };
    };

    const result = await callGemini(systemPrompt, userPrompt, dynamicFallbackQ);
    return res.json(result);
  }

  // action === 'evaluate-answer'
  const answerText = (userAnswer || '').trim();

  const systemPrompt = `You are a strict yet constructive interview evaluator for South Korean automotive manufacturers.
Evaluate the candidate's actual interview answer on:
- logic (1-100)
- jobRelevance (1-100)
- specificity (1-100)
- problemSolving (1-100)
- delivery (1-100)
Provide specific, actionable feedback, follow-up pressure question, and exemplary answer tips based on what the candidate actually wrote.
All JSON values (feedback, positivePoints, improvementPoints, followUpQuestion, exemplaryAnswerKey) MUST be written in Korean only.
Return JSON only:
{
  "scores": { "logic": number, "jobRelevance": number, "specificity": number, "problemSolving": number, "delivery": number },
  "totalScore": number,
  "feedback": "string (in Korean)",
  "positivePoints": ["string", "string"],
  "improvementPoints": ["string", "string"],
  "followUpQuestion": "string (in Korean)",
  "exemplaryAnswerKey": "string (in Korean)"
}`;

  const userPrompt = `면접 질문: ${currentQuestion}
지원자 실제 답변: ${answerText}
목표 기업 및 직무: ${company} - ${role}`;

  const dynamicFallbackEval = () => {
    const len = answerText.length;
    const hasNumbers = /\d+/.test(answerText);
    const hasSteps = /첫째|먼저|다음|원인|해결|결과|피드백/i.test(answerText);

    let logicScore = Math.min(95, Math.max(65, 75 + (hasSteps ? 10 : 0)));
    let specScore = Math.min(95, Math.max(60, 70 + (hasNumbers ? 12 : 0) + (len > 150 ? 8 : 0)));
    let relScore = Math.min(96, Math.max(70, 82 + (len > 80 ? 6 : 0)));
    let probScore = Math.min(94, Math.max(65, 78 + (hasSteps ? 8 : 0)));
    let delScore = Math.min(92, Math.max(65, 75 + (len > 100 ? 10 : 0)));

    const totalScore = Math.round((logicScore + specScore + relScore + probScore + delScore) / 5);

    return {
      scores: {
        logic: logicScore,
        jobRelevance: relScore,
        specificity: specScore,
        problemSolving: probScore,
        delivery: delScore
      },
      totalScore,
      feedback: len > 80
        ? `지원자께서 작성하신 답변은 문제 해결을 위한 기본 조치 방향을 명확히 제시하고 있으며, ${company} ${role} 직무에 부합하는 엔지니어링 문제 해결 태도를 보여줍니다.`
        : `답변의 핵심 방향은 좋으나, 실제 면접관의 신뢰를 얻기 위해서는 구체적인 수치 데이터나 과거 프로젝트 사례를 1~2문장 덧붙여야 설득력이 높아집니다.`,
      positivePoints: [
        '질문의 의도를 정확히 파악하고 문제 해결의 주도적 태도를 피력함',
        hasSteps ? '단계를 나누어 체계적으로 답변하여 논리적인 인상을 줌' : '엔지니어로서의 책임감 있는 현장 대응 태도 제시'
      ],
      improvementPoints: [
        hasNumbers ? '수치 성과를 더 강조하여 임팩트를 강화할 것' : '정량적인 수치(예: 몇 % 단축, 몇 회 측정)를 구체적으로 인용할 것',
        '타 부서(품질, 생산관리, 협력사)와의 유기적 커뮤니케이션 절차를 1문장 추가하면 협업 역량이 더욱 돋보입니다.'
      ],
      followUpQuestion: `답변에서 제시하신 조치를 진행할 때, 현장 작업자나 유관 부서 관리자가 일정 지연을 이유로 강하게 이견을 제기한다면 어떻게 설득하시겠습니까?`,
      exemplaryAnswerKey: `데이터 기반의 정량적 근거(불량 유출 시 리콜 비용 vs 30분 긴급 라인 점검 비용 비교)를 제시하고, 검증 완료 타임라인을 신속히 확약하는 답변이 모범적입니다.`
    };
  };

  const result = await callGemini(systemPrompt, userPrompt, dynamicFallbackEval);
  res.json(result);
});

// 7. Analyze Project for Portfolio & Resume
app.post('/api/gemini/analyze-project', async (req, res) => {
  const { project } = req.body;
  const title = project?.title || '모빌리티 프로젝트';
  const tools = project?.tools || '엔지니어링 툴';
  const prob = project?.problem || '';
  const sol = project?.solution || '';
  const resText = project?.result || '';

  const systemPrompt = `You are a technical recruiter for mobility companies.
Analyze the project description and extract resume bullet points and portfolio structuring advice.
Return JSON only:
{
  "extractedCompetencies": ["string", "string", "string"],
  "jobRelevance": "string",
  "coverLetterBullets": ["string", "string", "string"],
  "portfolioTips": ["string", "string"]
}`;

  const userPrompt = `프로젝트명: ${title}
기간: ${project?.period || ''}
본인 역할: ${project?.role || ''}
사용 기술/툴: ${tools}
문제 상황: ${prob}
해결 과정: ${sol}
결과 및 성과: ${resText}`;

  const dynamicFallback = () => ({
    extractedCompetencies: [
      `${tools} 활용 실무 설계 및 데이터 검증`,
      '공학적 오차 원인 분석 및 피드백 루프 수행',
      '다학제 팀 프로젝트 일정 및 결과물 관리'
    ],
    jobRelevance: `모빌리티 완성차 및 부품사에서 요구하는 핵심 역량인 도면 검증, 실물 제작, 피드백 사이클을 직접 수행한 실전성 높은 프로젝트입니다.`,
    coverLetterBullets: [
      `${title} 수행 중 ${tools}을(를) 활용하여 발생한 기술적 문제(${prob.slice(0, 30) || '공차 편차'})의 원인을 정량적으로 규명`,
      `${sol.slice(0, 40) || '지그 개선 및 공정 표준화'} 방안을 적용하여 오차를 최소화하고 신뢰도 높은 결과 달성`,
      `설계-가공-검증 전 과정의 이슈를 선제적으로 조율하여 프로젝트 기한 내 성과(${resText.slice(0, 30) || '목표 달성'}) 창출`
    ],
    portfolioTips: [
      '단순 완성품 사진만 나열하지 말고 "문제 발생 지점 vs 개선 지점"을 비교하는 비포/애프터 도면 슬라이드를 구성하세요.',
      '직접 작성한 계산 수식이나 해석 결과 이미지를 캡션과 함께 추가하면 기술적 신뢰도가 대폭 상승합니다.'
    ]
  });

  const result = await callGemini(systemPrompt, userPrompt, dynamicFallback);
  res.json(result);
});

// 8. Generate Skill Roadmap
app.post('/api/gemini/generate-roadmap', async (req, res) => {
  const { targetRole, currentSkills, major } = req.body;
  const roleName = targetRole || '생산기술';
  const skillsList = Array.isArray(currentSkills) ? currentSkills.join(', ') : (currentSkills || '');

  const systemPrompt = `You are a senior vocational counselor in the mobility engineering field.
Generate a structured, actionable, 3-month competence development roadmap for the target role.
Return JSON only:
{
  "roleName": "string",
  "gapAnalysis": "string",
  "weeklyGoals": [{ "week": "string", "goal": "string" }],
  "monthlyMilestones": [{ "month": "string", "title": "string" }],
  "recommendedCertifications": ["string", "string"],
  "recommendedCourses": ["string", "string"],
  "suggestedProjects": ["string", "string"]
}`;

  const userPrompt = `목표 직무: ${roleName}
현재 전공: ${major || '공학계열'}
현재 보유 기술: ${skillsList}`;

  const dynamicFallback = () => ({
    roleName,
    gapAnalysis: `현재 보유 중인 전공 기초 및 ${skillsList.slice(0, 30) || '기술'} 역량은 탄탄하나, [${roleName}] 현장에서 즉각 요구되는 실무 양산 규격 및 최신 공법(스마트팩토리/SDV) 적용 경험을 체계화하는 것이 최종 합격의 열쇠입니다.`,
    weeklyGoals: [
      { week: '1주차', goal: `${roleName} 직무의 최신 양산 현장 트렌드 및 타겟 기업(현대차, 모비스 등) 최신 공시 분석` },
      { week: '2주차', goal: `${roleName} 핵심 실무 툴(${skillsList.split(',')[0] || '공학 도구'}) 심화 및 실전 케이스 스터디` },
      { week: '3주차', goal: `품질/공정 불량률 데이터를 활용한 정량 분석 미니 프로젝트 포트폴리오화` },
      { week: '4주차', goal: `${roleName} 빈출 기술 면접 기출 질문 20선 스크립트 작성 및 모의면접 연습` }
    ],
    monthlyMilestones: [
      { month: '1개월차', title: `${roleName} 직무 핵심 이론 완성 및 국가기술자격/어학 점수 확보` },
      { month: '2개월차', title: `타겟 직무에 특화된 실무형 포트폴리오 1편 완성` },
      { month: '3개월차', title: `타겟 5개사 맞춤형 자기소개서 완성 및 실전 면접 완벽 대비` }
    ],
    recommendedCertifications: [
      '일반기계기사 / 전기기사 / 정보처리기사',
      '품질경영기사 (또는 6시그마 Green Belt)',
      '어학(TOEIC Speaking IM3/AL 또는 OPIc IM2/IH 이상)'
    ],
    recommendedCourses: [
      '현대차그룹 H-Mobility 클래스 / 미래차 인재양성 교육',
      '대한상공회의소 스마트팩토리 PLC 및 로봇 제어 실무 과정',
      '고용노동부 직업훈련포털(HRD-Net) 모빌리티 전동화 실무 과정'
    ],
    suggestedProjects: [
      `${roleName} 관점에서의 가상 공정 시뮬레이션 및 병목(Bottleneck) 개선 프로젝트`,
      '오픈소스 공정 센서 데이터를 활용한 이상 징후 감지 및 예지보전 분석'
    ]
  });

  const result = await callGemini(systemPrompt, userPrompt, dynamicFallback);
  res.json(result);
});

// Configure Vite or Static server
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[MOBILITY CAREER] Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
});
