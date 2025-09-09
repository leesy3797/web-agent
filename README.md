# Smart Web Automation Agent
## Human Like Web Action with LangGraph & Playwright-MCP-tool

인간의 Web Browser 탐색 방식을 모방하는 ReAct 기반의 AI Agent로 LangGraph와 MCP-Playwright를 활용하여 복잡한 Web 작업을 자동화합니다. 구조화된 에이전틱 워크플로우와 Human-in-the-Loop 패턴을 통해 안정적이고 신뢰할 수 있는 Web 자동화 기능을 제공합니다.

## 🎯 설계 철학

### 1. 구조화된 워크플로우
- **역할(Role) 기반 노드 분리**: 
  - 각 노드는 `계획수립, 실행, 보고`와 같은 *명확한 역할(Role)*과 *책임(Responsibility)*을 가집니다. 
  - 이로써 복잡한 태스크 처리 시에도 에이전트의 행동이 더욱 예측 가능해지고, 문제가 발생했을 때 특정 노드를 집중적으로 디버깅하기 쉬워집니다.

- **`AgentState`를 통한 상태 관리**: 
  - 단순히 대화 기록(Messages)에만 의존하는 것이 아닌, 에이전트와의 여러 상호 작용 결과(예: extracted_data, last_snapshot)를 저장하고 추적합니다. 
  - 이러한 상태 관리는 고도화된 Context Engineering으로 이어져 에이전트의 정교한 웹 작업 수행이 가능해집니다. 

### 2. Human-in-the-Loop 패턴

- **Task 명확화 단계**: 
  - 에이전트는 사용자의 초기 요청을 분석하여 웹 자동화에 필요한 정보가 불충분하다고 판단할 경우, 구체적인 질문을 통해 추가 정보를 확보합니다. (예: 선호하는 웹사이트, 사용자의 취향 등)
  - 사용자가 제공하기 어려운 부분은 에이전트가 자율적으로 판단하여 성공적인 작업 수행이 가능하도록 정보를 확장합니다. 

- **Plan 수립 및 승인**: 
  - 에이전트가 수립한 실행 계획은 사용자의 검토와 명시적인 승인을 거친 후에만 실행됩니다. 
  - 사용자는 계획을 거부하고 직접 피드백을 제공함으로써, 에이전트가 계획을 수정하고 최적화하도록 개입할 수 있습니다. 

### 3. 에이전트 최적화

* **스냅샷 별도 관리**:
    * 대용량 HTML 스냅샷이 `messages` 필드에 누적되어 발생하는 `MAX_TOKENS` 이슈와 메모리 낭비를 방지합니다. 
    - `browser_snapshot` 실행 결과는 `last_snapshot`이라는 별도 필드에서 관리합니다.
* **동적 컨텍스트 관리**:
    * 에이전트의 작업이 길어질 때 컨텍스트 윈도우를 초과하여 발생하는 `MAX_TOKENS` 오류에 선제적으로 대응합니다. 
    - 에러 발생 시 에이전트에게 제공되는 `messages`의 길이를 점진적으로 축소하여, 에이전트가 도중에 중단 없이 태스크를 완수할 수 있도록 합니다. 


## 🏗️ 아키텍처

### LangGraph 워크플로우

```
START → Clarify → Planning → Human Approval → Agent → Tools → Report → END
```

### 노드별 상세 설계

---

#### 1. Clarify 노드
* **입력**: 사용자의 초기 요청
* **처리**: LLM을 통한 자동화에 필요한 정보 누락 여부 판단
* **출력**:
    * `need_clarification = True`: 추가 질문
    * `need_clarification = False`: 계획 단계로 진행 메시지
* **라우팅**: `get_command_destination`을 통한 조건부 분기
    * 질문 요청 → `clarify` 노드(루프)
    * 정보 충분 → `planning` 노드
    * 기타 → `END` 노드

---

#### 2. Planning 노드
* **입력**: 사용자와의 대화 기록
* **처리**: 대화 내용을 바탕으로 상세 실행 계획 수립
* **출력**: Plan 스키마에 맞춘 `List[str]` 타입의 계획
* **상태**: `plan` 필드 업데이트
* **라우팅**: `Human Approval` 노드로 이동

---

#### 3. Human Approval 노드
* **입력**: 생성된 계획
* **처리**: 사용자에게 계획을 제시하고 승인 요청
* **출력**: 사용자 승인 요청(yes/no)
* **라우팅**: `get_command_destination`을 통한 분기
    * `yes` → `Agent` 노드
    * `no` → 피드백 수집 후 `Planning` 노드
    * 기타 → 종료 여부 확인

---

#### 4. Agent 노드
* **입력**: `browser_snapshot` 결과를 제외한 `messages` (마지막 메시지가 스냅샷일 경우 포함)
* **처리**: 현재 계획 단계에 따라 필요한 도구 호출 결정
* **출력**: `tool_calls` 결과
* **라우팅**: `should_continue`를 통한 조건부 분기
    * `tool_calls` 존재 → `Tool` 노드
    * 계획 단계 남음 → `Agent` 노드(루프)
    * 계획 완료 → `Report` 노드
* **에러 핸들링**: `MAX_TOKENS` 이슈 시 메시지 점진적 축소

---

#### 5. Tool 노드
* **입력**: `state`와 `tool_executor`
* **처리**: 마지막 메시지의 `tool_calls` 실행
* **출력**: 도구별 반환 결과
    * `think_tool`: `result`, `current_plan_step`
    * `mcp_playwright_tool`: `result`, `none`
* **특수 처리**:
    * `browser_snapshot`: `last_snapshot` 필드 업데이트
    * `extracted_data_tool`: `extracted_data` 필드 업데이트

---

#### 6. Report 노드
* **입력**: `AgentState`(plan, extracted_data 등)
* **처리**: 실행된 계획과 추출된 데이터를 바탕으로 최종 보고서 생성
* **출력**: `final_answer`, `workflow_summary`

---

#### 7. END 노드
* **입력**: `AgentState`
* **처리**: 사용자 요청 유형에 따른 결과 출력
    * 웹 작업만 요청: `workflow_summary`
    * 결과물 요청: `final_answer` + `workflow_summary`

### Agent State 구조

`AgentState`는 전체 워크플로우의 상태를 관리하는 핵심 객체입니다.
```python
class AgentState:
    task: str                         # 사용자 요청 (원본 태스크)
    messages: Annotated[List[AnyMessage], operator.add] # 대화 기록 (누적)
    plan: List[str]                   # 실행 계획 (단계별)
    current_plan_step: int            # 현재 실행 중인 계획 단계
    action_history: Annotated[List[str], operator.add] # 실행된 액션 기록
    last_error: Optional[str]         # 마지막 발생한 오류
    extracted_data: Annotated[Dict[str, Any], operator.ior] # 추출된 데이터 (누적)
    final_answer: str                 # 최종 답변
    workflow_summary: str             # 워크플로우 요약
    max_messages_for_agent: Optional[int] # 컨텍스트 관리
    last_snapshot: Optional[str] = None # 브라우저 스냅샷 (메모리 최적화)
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

- **LangGraph**: 에이전트 워크플로우 그래프 생성 및 오케스트레이션 
- **LangChain**: LLM 제공자들의 API를 통합된 형태로 관리하는 프레임워크
- **langchain-mcp-adapters**: MCP 툴을 Langchain 툴로 변환하는 wrapper 라이브러리
- **MCP-Playwright**: Model Context Protocol을 통한 Playwright 브라우저 자동화
- **Google Gemini 2.0 Flash**: Smart Web Automation Agent의 메인 LLM

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


### (1) 도구 시스템

* **MCP-Playwright 도구**: `browser_navigate`, `browser_click`, `browser_type`, `browser_snapshot` 등 Playwright 기반의 웹 상호작용을 위한 도구 모음입니다.
* **커스텀 도구**:
    * `extracted_data_tool`: 웹페이지에서 추출한 키-값 형태의 데이터를 `extracted_data` 필드에 저장합니다.
    * `finish_step`: 현재 계획 단계의 목표가 달성되었을 때, 다음 단계로 진행하도록 에이전트에게 신호를 보냅니다.

### (2) 프롬프트 시스템

* **Clarification Prompt**: 사용자 요청의 완성도를 평가하고, 누락된 정보를 식별하여 질문을 생성합니다.
* **Planning Prompt**: 대화 내용을 바탕으로 고수준의 실행 계획을 수립합니다.
* **Command Prompt**: 에이전트의 행동 지침을 정의하고, 도구 사용 가이드라인 및 현재 계획 단계에 대한 컨텍스트를 제공합니다.


### (3) 상태 관리 시스템

- 자동 저장 메커니즘
  - **실시간 저장**: 각 노드 실행 후 AgentState 자동 저장
  - **JSON 직렬화**: 메시지 객체를 JSON으로 변환하여 저장
  - **세션 복구**: 이전 작업 상태를 완전히 복원

- 컨텍스트 관리
  - **동적 메시지 축소**: MAX_TOKENS 이슈 시 메시지 수를 점진적으로 감소
  - **스냅샷 분리**: 브라우저 스냅샷을 별도 필드로 관리하여 메모리 최적화
  - **에러 복구**: 토큰 제한 해결 후 메시지 수 자동 복구

---

## 🔧 기술적 특징
### 메모리 최적화
- **스냅샷 분리**: 브라우저 스냅샷을 `last_snapshot` 필드로 분리하여 메시지 히스토리 최적화
- **점진적 컨텍스트 축소**: MAX_TOKENS 이슈 발생 시 메시지 수를 5→4→3→2→1로 점진적 감소
- **자동 복구**: 토큰 제한 해결 후 메시지 수를 5로 자동 복구

### 에러 처리
- **재귀 제한**: GraphRecursionError 발생 시 사용자에게 진행 상황 요약 제공
- **도구 실행 오류**: 개별 도구 실행 실패 시에도 워크플로우 계속 진행
- **상태 복구**: 오류 발생 시에도 AgentState는 보존되어 세션 복구 가능
- **Rate Limiting**: 0.25 requests/second로 제한하여 API 과부하 방지

### 🔄 개발 현황 및 로드맵
---
### 현재 구현 완료
* ✅ LangGraph 워크플로우 오케스트레이션
* ✅ MCP-Playwright 통합
* ✅ Human-in-the-Loop 패턴 구현 (clarify노드 + human_approval)
* ✅ 상태 관리 및 세션 복구
* ✅ MAX_TOKENS 이슈 대응 및 동적 메시지 축소
* ✅ 스냅샷 분리 최적화 (`last_snapshot` 필드)

### 개발 중 (프로젝트 기술서 기준)
* 🔄 **스냅샷 결과 처리**: `browser_snapshot` 결과를 `last_snapshot` 필드로 관리하여 메모리 최적화
* 🔄 **extracted_data_tool**: 중간 결과 저장을 통해 데이터 누적
* 🔄 **Report 노드**: 최종 결과 보고 단계 노드 구현
* 🔄 **코드 리팩토링**: Node, Tool, Agent, Utils로 코드 리팩토링

### 향후 계획
* 📋 **IP차단 문제 해소**: 과도한 요청으로 인한 IP 차단 방지를 위한 처리
* 📋 **State 분리**: Clarify/Planning 단계와 Agent/Tool/Report 단계의 상태 분리 검토 -> PrepareState와 AgentState
* 📋 **Human-in-the-Loop 고도화**: 에이전트의 `실행 단계(Agent노드+Tool노드)`에서 웹 작업 중 필요한 추가 정보를 HITL로 인간 개입
* 📋 **Report 노드 QA**: Report노드의 실행 결과에 대한 QA
* 📋 **웹 인터페이스**: Web UI 형태로 브라우저 상호 작용(좌)과 에이전트의 액션(우)를 프론트로 구현
---

**Web Automation Agent** - AI 기반 웹 자동화의 새로운 패러다임 🤖🌐
