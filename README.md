# Web Automation Agent

AI 기반 웹 자동화 에이전트로, LangGraph와 MCP-Playwright를 활용하여 복잡한 웹 작업을 자동화합니다. 이 프로젝트는 구조화된 워크플로우와 Human-in-the-Loop 패턴을 통해 안정적이고 신뢰할 수 있는 웹 자동화를 제공합니다.

## 🎯 설계 철학

### 1. 구조화된 워크플로우
- **명확한 단계 분리**: 각 노드는 명확한 책임과 입출력을 가짐
- **상태 기반 관리**: AgentState를 통한 일관된 상태 추적
- **에러 복구**: MAX_TOKENS 이슈 대응 및 점진적 컨텍스트 축소

### 2. Human-in-the-Loop 패턴
- **명확화 단계**: 불완전한 요청에 대한 추가 정보 수집
- **계획 승인**: 사용자가 생성된 계획을 검토하고 승인
- **피드백 루프**: 거부된 계획에 대한 피드백 수집 및 재계획

### 3. 메모리 효율성
- **스냅샷 분리**: 브라우저 스냅샷을 별도 필드로 관리하여 메시지 히스토리 최적화
- **컨텍스트 관리**: 동적 메시지 축소를 통한 토큰 제한 대응

## 🏗️ 아키텍처

### LangGraph 워크플로우

```
START → Clarify → Planning → Human Approval → Agent → Tools → Report → END
```

### 노드별 상세 설계

#### 1. Clarify 노드
- **입력**: 사용자의 초기 요청 (task)
- **처리**: LLM이 자동화 처리에 필요한 정보가 누락되었는지 판단
- **출력**: 
  - `need_clarification = True`: 추가 질문
  - `need_clarification = False`: 계획 단계로 진행 메시지
- **라우팅**: `get_command_destination`을 통한 3가지 분기
  - 질문 요청 → clarify 노드 (루프)
  - 정보 충분 → planning 노드
  - 기타 → END 노드

#### 2. Planning 노드
- **입력**: 사용자와의 대화 기록 (초기 요청 + 추가 정보)
- **처리**: 대화 내용을 바탕으로 상세한 실행 계획 수립
- **출력**: Plan 스키마에 맞춘 `List[str]` 타입의 계획
- **상태**: plan 필드 업데이트
- **라우팅**: Human Approval 노드로 이동

#### 3. Human Approval 노드
- **입력**: 사용자와의 대화 기록
- **처리**: 생성된 계획을 사용자에게 제시하고 승인 요청
- **출력**: 사용자 승인 요청 (yes/no)
- **라우팅**: `get_command_destination`을 통한 분기
  - `yes` → Agent 노드
  - `no` → 피드백 수집 후 Planning 노드
  - 기타 → 종료 여부 확인

#### 4. Agent 노드
- **입력**: 
  - 일반: browser_snapshot 결과를 제외한 messages
  - 특수: 마지막 메시지가 browser_snapshot인 경우 해당 결과 포함
- **처리**: 계획의 현재 단계에 따라 필요한 도구 호출 결정
- **출력**: tool_calls 결과
- **라우팅**: `should_continue`를 통한 3가지 분기
  - tool_calls 존재 → Tool 노드
  - 계획 단계 남음 → Agent 노드 (루프)
  - 계획 완료 → Report 노드
- **에러 핸들링**: MAX_TOKENS 이슈 시 점진적 메시지 축소

#### 5. Tool 노드
- **입력**: state와 tool_executor
- **처리**: 마지막 메시지의 tool_calls를 tool_executor로 실행
- **출력**: 도구별 반환 결과
  - `think_tool`: result, current_plan_step
  - `mcp_playwright_tool`: result, none
- **특수 처리**:
  - `browser_snapshot`: last_snapshot 필드 업데이트
  - `extracted_data_tool`: extracted_data 필드 업데이트

#### 6. Report 노드
- **입력**: AgentState (plan, extracted_data 등)
- **처리**: 실행된 계획과 추출된 데이터를 바탕으로 최종 보고서 생성
- **출력**: final_answer, workflow_summary

#### 7. END 노드
- **입력**: AgentState
- **처리**: 사용자 요청 유형에 따른 결과 출력
  - 웹 작업만 요청: workflow_summary
  - 결과물 요청: final_answer + workflow_summary

### Agent State 구조

```python
class AgentState:
    # 사용자 요청 (원본 태스크)
    task: str
    
    # 대화 기록 (누적)
    messages: Annotated[List[AnyMessage], operator.add]
    
    # 실행 계획 (단계별)
    plan: List[str]
    
    # 현재 실행 중인 계획 단계 (0-based)
    current_plan_step: int
    
    # 실행된 액션 기록 (누적)
    action_history: Annotated[List[str], operator.add]
    
    # 마지막 발생한 오류
    last_error: Optional[str]
    
    # 추출된 데이터 (누적)
    extracted_data: Annotated[Dict[str, Any], operator.ior]
    
    # 최종 답변
    final_answer: str
    
    # 워크플로우 요약
    workflow_summary: str
    
    # 컨텍스트 관리 (MAX_TOKENS 대응)
    max_messages_for_agent: Optional[int] = 5
    
    # 브라우저 스냅샷 (메모리 최적화)
    last_snapshot: Optional[str] = None
```

### 스키마 정의

#### Clarification 스키마
```python
class Clarification(BaseModel):
    need_clarification: bool    # 추가 정보 필요 여부
    question: str              # 사용자에게 할 질문
    message_to_user: str       # 계획 진행 메시지
```

#### Plan 스키마
```python
class Plan(BaseModel):
    steps: List[str]           # 단계별 실행 계획
```

## 🛠️ 기술 스택

- **LangGraph**: 워크플로우 오케스트레이션 및 상태 관리
- **LangChain**: LLM 통합 및 도구 관리 프레임워크
- **MCP-Playwright**: Model Context Protocol을 통한 Playwright 브라우저 자동화
- **Google Gemini 2.0 Flash**: 언어 모델 (구조화된 출력 지원)
- **Rich**: 터미널 UI 및 로깅
- **MCP (Model Context Protocol)**: 도구 통합 표준

## 📦 설치 및 설정

```bash
# 1. 저장소 클론 및 의존성 설치
git clone <repository-url> && cd web-automation
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 환경 변수 설정
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env

# 3. MCP-Playwright 브라우저 설치
npx playwright install

# 4. 실행
python app.py
```

## 🚀 사용법

### 기본 워크플로우

1. **요청 입력**: 자연어로 웹 자동화 작업 요청
2. **명확화**: 필요한 경우 추가 정보 수집
3. **계획 승인**: 생성된 계획 검토 및 승인
4. **자동 실행**: AI 에이전트가 계획에 따라 웹 작업 수행
5. **결과 보고**: 최종 결과 및 작업 요약 제공

### 세션 관리

- **자동 저장**: 작업 진행 상황이 `agent_state.json`에 자동 저장
- **세션 복구**: 이전 작업을 이어서 진행 가능
- **상태 추적**: 실시간으로 작업 진행 상황 모니터링

## 📁 프로젝트 구조

```
web-automation/
├── app.py                           # 메인 애플리케이션 (워크플로우 오케스트레이션)
├── requirements.txt                 # Python 의존성
├── .env                            # 환경 변수 (Google AI API 키)
├── .gitignore                      # Git 무시 파일
├── src/                            # 소스 코드
│   ├── state_automation.py         # AgentState 정의 및 Pydantic 스키마
│   ├── web_automation_planning.py  # Clarify, Planning, Human Approval 노드
│   ├── prompts.py                  # 프롬프트 템플릿 (Clarification, Planning, Command)
│   └── utils.py                    # 유틸리티 함수 (메시지 포맷팅, 디스플레이)
├── browser_data/                   # MCP-Playwright 브라우저 데이터 (자동 생성)
├── agent_state.json                # 에이전트 상태 저장 (세션 복구용)
└── README.md                       # 프로젝트 문서
```

## 🔧 주요 컴포넌트

### 도구 시스템 (Tools)

#### MCP-Playwright 도구
- **browser_navigate**: 웹페이지 탐색
- **browser_click**: 요소 클릭
- **browser_type**: 텍스트 입력
- **browser_snapshot**: 페이지 스냅샷 (HTML 파싱)
- **browser_wait**: 요소 대기
- **browser_scroll**: 페이지 스크롤

#### 커스텀 도구
- **extracted_data_tool**: 웹페이지에서 추출한 데이터를 AgentState에 저장
- **finish_step**: 현재 계획 단계 완료 표시 및 다음 단계로 진행

### 프롬프트 시스템

#### Clarification Prompt
- 사용자 요청의 완성도 평가
- 누락된 정보 식별 및 질문 생성
- 구조화된 출력 (Clarification 스키마)

#### Planning Prompt
- 대화 내용을 바탕으로 실행 계획 수립
- 단계별 작업 분해
- 구조화된 출력 (Plan 스키마)

#### Command Prompt
- 에이전트의 행동 지침
- 도구 사용 가이드라인
- 현재 계획 단계에 대한 컨텍스트

### 상태 관리 시스템

#### 자동 저장 메커니즘
- **실시간 저장**: 각 노드 실행 후 AgentState 자동 저장
- **JSON 직렬화**: 메시지 객체를 JSON으로 변환하여 저장
- **세션 복구**: 이전 작업 상태를 완전히 복원

#### 컨텍스트 관리
- **동적 메시지 축소**: MAX_TOKENS 이슈 시 메시지 수를 점진적으로 감소
- **스냅샷 분리**: 브라우저 스냅샷을 별도 필드로 관리하여 메모리 최적화
- **에러 복구**: 토큰 제한 해결 후 메시지 수 자동 복구

## 🎯 사용 사례

| 카테고리 | 사용 사례 | 설명 | 예시 |
|---------|----------|------|------|
| **웹 스크래핑** | 제품 정보 수집 | 온라인 쇼핑몰에서 상품 정보 추출 | "쿠팡에서 아이폰 15 가격과 리뷰 수집해줘" |
| | 가격 비교 | 여러 사이트의 가격 정보 비교 | "다나와에서 노트북 가격 비교해줘" |
| | 뉴스 수집 | 뉴스 사이트에서 특정 키워드 검색 | "네이버 뉴스에서 AI 관련 기사 수집해줘" |
| **폼 자동화** | 설문조사 작성 | 온라인 설문조사 자동 작성 | "구글 폼에 설문조사 답변 입력해줘" |
| | 회원가입 | 웹사이트 회원가입 프로세스 자동화 | "네이버 회원가입 도와줘" |
| | 데이터 입력 | 반복적인 데이터 입력 작업 | "엑셀 데이터를 웹 폼에 일괄 입력해줘" |
| **웹 테스팅** | UI 테스트 | 웹 애플리케이션 UI 기능 검증 | "로그인 기능이 정상 작동하는지 테스트해줘" |
| | 기능 검증 | 특정 기능의 동작 확인 | "장바구니 추가 기능 테스트해줘" |
| | 성능 모니터링 | 웹사이트 로딩 속도 측정 | "페이지 로딩 시간 측정해줘" |
| **데이터 수집** | 시장 조사 | 특정 시장의 동향 파악 | "부동산 시장 동향 조사해줘" |
| | 경쟁사 분석 | 경쟁사 웹사이트 분석 | "경쟁사 제품 정보 수집해줘" |
| | 트렌드 파악 | 소셜미디어나 뉴스 트렌드 분석 | "최신 기술 트렌드 조사해줘" |

## 🔧 기술적 특징

### 메모리 최적화
- **스냅샷 분리**: 브라우저 스냅샷을 `last_snapshot` 필드로 분리하여 메시지 히스토리 최적화
- **점진적 컨텍스트 축소**: MAX_TOKENS 이슈 발생 시 메시지 수를 5→4→3→2→1로 점진적 감소
- **자동 복구**: 토큰 제한 해결 후 메시지 수를 5로 자동 복구

### 에러 처리
- **재귀 제한**: GraphRecursionError 발생 시 사용자에게 진행 상황 요약 제공
- **도구 실행 오류**: 개별 도구 실행 실패 시에도 워크플로우 계속 진행
- **상태 복구**: 오류 발생 시에도 AgentState는 보존되어 세션 복구 가능

### Human-in-the-Loop
- **명확화 루프**: 불완전한 요청에 대한 반복적 질문
- **계획 승인**: 생성된 계획에 대한 사용자 검토 및 승인
- **피드백 루프**: 거부된 계획에 대한 피드백 수집 및 재계획

## ⚠️ 주의사항 및 제한사항

### API 제한
- **Google AI API**: 사용량 제한 및 비용 고려
- **Rate Limiting**: 0.25 requests/second로 제한하여 API 과부하 방지

### 브라우저 리소스
- **메모리 사용량**: 브라우저 세션 지속으로 인한 메모리 사용량 증가
- **세션 관리**: MCP-Playwright 세션 재사용으로 리소스 효율성 확보

### 웹사이트 정책
- **이용약관 준수**: 대상 웹사이트의 로봇 배제 표준(robots.txt) 및 이용약관 준수
- **접근 빈도**: 과도한 요청으로 인한 IP 차단 방지

### 데이터 보안
- **민감 정보**: 개인정보나 인증 정보 처리 시 주의
- **상태 저장**: `agent_state.json`에 민감한 정보가 저장될 수 있음

## 🔄 개발 현황 및 로드맵

### 현재 구현 완료
- ✅ LangGraph 워크플로우 오케스트레이션
- ✅ MCP-Playwright 통합
- ✅ Human-in-the-Loop 패턴
- ✅ 상태 관리 및 세션 복구
- ✅ MAX_TOKENS 이슈 대응
- ✅ 스냅샷 분리 최적화

### 개발 중 (프로젝트 기술서 기준)
- 🔄 **스냅샷 결과 처리**: `last_snapshot` 필드를 통한 메모리 최적화
- 🔄 **extracted_data_tool**: 중간 결과 저장을 통한 데이터 누적
- 🔄 **Report 노드**: 최종 결과 보고서 생성

### 향후 계획
- 📋 **State 분리**: Clarify/Planning 단계와 Agent/Tool/Report 단계의 상태 분리 검토
- 📋 **Human-in-the-Loop 고도화**: 일관된 인간 개입 처리 방식 통일
- 📋 **다중 브라우저 지원**: 동시 다중 브라우저 세션 관리
- 📋 **스케줄링 기능**: 정기적 웹 자동화 작업 스케줄링
- 📋 **웹 인터페이스**: 브라우저 기반 사용자 인터페이스
- 📋 **플러그인 시스템**: 확장 가능한 도구 및 노드 시스템

## 🤝 기여하기

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 지원 및 문의

- **이슈 리포트**: [GitHub Issues](https://github.com/your-repo/issues)
- **기능 요청**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **문서**: 이 README.md 파일 참조

---

**Web Automation Agent** - AI 기반 웹 자동화의 새로운 패러다임 🤖🌐
