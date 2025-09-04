import operator
import asyncio
import uuid
from typing import TypedDict, Annotated, List
from functools import partial

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

import os
import logging
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError
from langchain_core.tools import tool
from langgraph.types import Command

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from dotenv import load_dotenv

from src.prompts import command_prompt
from src.web_automation_planning import clarify_with_user, planning_automation, human_approval
from src.state_automation import AgentState
import json 

load_dotenv()
logging.basicConfig(level=logging.ERROR)

# --- 1. 초기 설정 ---
console = Console()
STATE_FILE = "agent_state.json"

mcp_config = {
    "playwright": {
        "command": "npx",
        "args": [
            "-y",
            "@playwright/mcp@latest",
            # "--user-data-dir=./browser_data",  # 👈 세션 유지
            # "--save-session"                  # 👈 세션 저장
        ],
        "transport": "stdio"
    }
}

_client = None

async def get_mcp_client() -> MultiServerMCPClient:
    """
    전역 MCP Client 생성 및 재사용.
    최초 실행 시에만 서버를 띄우고, 이후에는 같은 세션을 재사용.
    """
    global _client
    if _client is None:
        console.print(Panel("[bold yellow]Creating MCP client...[/bold yellow]", expand=False))
        _client = MultiServerMCPClient(mcp_config)
        console.print("[green]✓ MCP client started successfully![/green]")
    return _client


@tool(parse_docstring=True)
def think_tool(reflection: str, current_plan_step: int) -> str:
    """
    Tool for strategically reflecting on the progress of a web automation task and deciding the next action.

    Args:
        reflection: Your detailed thoughts on the webpage state, task progress, obstacles, and the next action plan.
                    You should structure your reflection to include your observation, analysis, and a proposed next step.
        current_plan_step: The index (0-based) of the current step in the plan you are working on.
    
    Returns:
        A confirmation that the reflection has been recorded to inform the next decision.
    """
    console.print(Panel(f"[bold yellow]🤔 Agent's Reflection:[/bold yellow]\n{reflection}", border_style="yellow"))
    return f"Reflection recorded. Currently on plan step {current_plan_step}."


# --- 3. 툴 세팅 ---
async def setup_tools():
    client = await get_mcp_client()
    console.print(Panel("[bold yellow]Getting tools....[/bold yellow]", expand=False))
    mcp_tools = await client.get_tools()   # 동일 세션에서 툴 로드
    all_tools = mcp_tools + [think_tool]
    return all_tools

def show_tools_table(all_tools) -> List:
    table = Table(title="Available Agent Tools", show_header=True, header_style="bold magenta")
    table.add_column("Tool Name", style="cyan", width=25)
    table.add_column("Description", style="white", width=80)
    for t in all_tools:
        description = t.description[:77] + "..." if len(t.description) > 80 else t.description
        table.add_row(t.name, description)
    console.print(table)
    console.print(f"[bold green]✓ Successfully retrieved {len(all_tools)} tools[/bold green]")

# --- 상태 저장/불러오기 헬퍼 함수 ---
def save_agent_state(state: AgentState, filename: str = "agent_state.json"):
    """
    AgentState 객체를 JSON 직렬화 가능한 형태로 변환하여 파일로 저장합니다.
    """
    try:
        # AgentState에서 JSON 직렬화 불가능한 객체를 변환
        serializable_state = {
            "task": state.get("task"),
            # LangChain 메시지 객체를 dict 형태로 변환
            "messages": [m.model_dump() for m in state.get("messages", [])],
            "plan": state.get("plan"),\
            "current_plan_step": state.get("current_plan_step"),
            "action_history": state.get("action_history"),
            "last_error": state.get("last_error"),
            "extracted_data": state.get("extracted_data"),
            "final_answer": state.get("final_answer"),
            "workflow_summary": state.get("workflow_summary")
        }
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=4, ensure_ascii=False)
        # print(f"Agent state successfully saved to {filename}")
    except TypeError as e:
        print(f"Error saving state: {e}")


def load_agent_state() -> AgentState or None:
    """JSON 파일에서 AgentState를 불러오기"""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding='utf-8') as f:
            loaded_data = json.load(f)

            messages = []
            for msg_data in loaded_data.get("messages", []):
                msg_type = msg_data.get("type")
                if msg_type == "human":
                    messages.append(HumanMessage(**msg_data))
                elif msg_type == "ai":
                    messages.append(AIMessage(**msg_data))
                elif msg_type == "tool":
                    messages.append(ToolMessage(**msg_data))
                elif msg_type == "system":
                    messages.append(SystemMessage(**msg_data))

            loaded_data["messages"] = messages
            console.print(Panel("[bold green]✓ 이전 작업 상태를 성공적으로 불러왔습니다.[/bold green]", border_style="green"))
            return loaded_data
    except Exception as e:
        console.print(Panel(f"[bold red]Error loading state: {e}. 새 작업을 시작합니다.[/bold red]", border_style="red"))
        return None
    
# Add this function to app.py
def summarize_agent_state(state: AgentState) -> str:
    """
    Generates a user-facing summary of the agent's current state.
    """
    summary = "Here is a summary of the task's progress so far:\n\n"

    task = state.get("task", "N/A")
    summary += f"**Task:** {task}\n\n"
    # 1. Summarize the plan
    plan_steps = state.get("plan", [])
    if plan_steps:
        summary += "**Current Plan:**\n"
        for i, step in enumerate(plan_steps):
            summary += f"- {step}\n"
    else:
        summary += "**Current Plan:** None\n"

    # 2. Summarize the action history (showing only the last 5 actions for clarity)
    action_history = state.get("action_history", [])
    if action_history:
        summary += "\n**Recent Web Interaction History:**\n"
        for action in action_history[-5:]:
            summary += f"- {action}\n"
    else:
        summary += "\n**Web Interaction History:** None\n"
    
    # 3. Summarize the extracted data
    extracted_data = state.get("extracted_data", {})
    if extracted_data:
        summary += "\n**Information Found So Far:**\n"
        for key, value in extracted_data.items():
            summary += f"- {key}: {value}\n"
    else:
        summary += "\n**Information Found So Far:** None\n"

    return summary

# --- 2. 도구 실행기 정의 ---
class ToolExecutor:
    def __init__(self, tools: List):
        self.tools_by_name = {tool.name: tool for tool in tools}

    async def __call__(self, tool_call: dict, config=None):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        console.print(Panel(f"[bold blue]🛠️ Executing Tool: `{tool_name}`[/bold blue]\nArguments: {tool_args}", border_style="blue"))
        if tool_name == "think_tool":
            result = self.tools_by_name[tool_name].invoke(tool_args)
            return result, tool_args.get("current_plan_step", 0)
        else:
            result = await self.tools_by_name[tool_name].ainvoke(tool_args)
            return result, None


# --- 4. LangGraph 워크플로우 정의 ---
# class AgentState(TypedDict): 
#     messages: Annotated[list[AnyMessage], operator.add]


async def agent_node(state: AgentState, model_with_tools):
    plan = state.get("plan", [])
    step_index = state.get("current_plan_step", 0)
    if plan and step_index < len(plan):
        console.print(
            Panel(
                f"Current Step ({step_index + 1}/{len(plan)}): [bold cyan]{plan[step_index]}[/bold cyan]", 
                title = "🚀 Executing Plan", 
                border_style="cyan"
            )
        )
    guidance = (
        "You must strictly follow the current plan step only.\n"
        "- Current plan index (0-based): {}\n"
        "- Current plan step: {}\n"
        "- Full plan (for reference only, do NOT execute future steps now):\n{}\n\n"
        "Constraints:\n"
        "1) Only perform actions necessary to achieve the deliverable of the CURRENT step.\n"
        "2) Do NOT pre-emptively execute later steps.\n"
        "3) After any browser action, call browser_snapshot, then call think_tool to reflect.\n"
        "4) When the CURRENT step's deliverable is satisfied, use think_tool and set current_plan_step to the next index.\n"
        "5) If the page state is unclear, prefer observation (browser_snapshot) before action.\n"
    ).format(
        int(step_index),
        plan[step_index] if plan and step_index < len(plan) else "N/A",
        "\n".join([f"  - {i+1}. {s}" for i, s in enumerate(plan)])
    )

    model_input_messages = list(state["messages"]) + [SystemMessage(content=guidance)]
    response = await model_with_tools.ainvoke(model_input_messages)
    return {"messages": [response]}


async def tool_node(state: AgentState, tool_executor: ToolExecutor):
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    last_error = None 
    current_plan_step = state.get("current_plan_step", 0)

    for tool_call in tool_calls:
        try:
            output, new_step = await tool_executor(tool_call)
            if new_step is not None:
                current_plan_step = new_step
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        except Exception as e:
            error_message = f"Error during tool execution: {e}"
            console.print(Panel(f"[bold red]{error_message}[/bold red]", border_style="red"))
            if "Timeout" in str(e):
                error_message += "\nSuggestion: The page might be slow to load. Consider using 'browser_wait_for' or retrying."
            elif "not found" in str(e) or "no element" in str(e):
                error_message += "\nSuggestion: The selector might be incorrect. Use 'browser_snapshot' to re-evaluate the page structure."
            tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call["id"]))
            last_error = error_message
    return {"messages": tool_messages, "last_error": last_error, "current_plan_step": int(current_plan_step)}


def should_continue(state: AgentState):
    if not state["messages"][-1].tool_calls:
        return "end"
    else:
        return "continue"

def get_command_destination(node_result) -> str:
    # node_result는 노드의 반환값이며, Command일 경우 해당 goto로 라우팅
    if isinstance(node_result, Command):
        return node_result.goto
    return "end"

# --- 7. 메인 실행 ---
async def main():

    client = await get_mcp_client()
    async with client.session("playwright") as session:
        mcp_tools = await load_mcp_tools(session)
        all_tools = mcp_tools + [think_tool]
        # all_tools = await setup_tools()
        show_tools_table(all_tools)
        tool_executor = ToolExecutor(all_tools)

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.25,
            check_every_n_seconds=0.1,
            max_bucket_size=5
        )
        console.print(Panel("[bold cyan]Rate limiter configured (~1 request / 4 sec)[/bold cyan]"))

        model = init_chat_model(
            model="google_genai:gemini-2.0-flash",
            rate_limiter=rate_limiter,
            temperature=0
        )
        model_with_tools = model.bind_tools(all_tools)

        workflow = StateGraph(AgentState)
        workflow.add_node("clarify_with_user", clarify_with_user)
        workflow.add_node("planning_automation", planning_automation)
        workflow.add_node("human_approval", human_approval)
        workflow.add_node("agent", partial(agent_node, model_with_tools=model_with_tools))
        workflow.add_node("tools", partial(tool_node, tool_executor=tool_executor))
        
        # workflow.add_edge(START, "clarify_with_user")
        workflow.set_entry_point("clarify_with_user")
        workflow.add_conditional_edges(
            "clarify_with_user",
            get_command_destination,
            {
                "clarify_with_user": "clarify_with_user",   
                "planning_automation": "planning_automation",
                "end" : END
            }
        )
        workflow.add_edge("planning_automation", "human_approval")

        workflow.add_conditional_edges(
            "human_approval",
            get_command_destination,
            {
                "agent" : "agent",
                "planning_automation": "planning_automation",
                "human_approval": "human_approval",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "agent", 
            should_continue, 
            {
                "continue": "tools", 
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        app = workflow.compile()
        console.print(Panel("[bold green]Web Automation Agent is ready.[/bold green]", title="🤖 Agent Ready"))

        current_state = load_agent_state()
        if current_state:
            console.print(Panel("[bold]이전 작업 내용:[/bold]\n[dim]Task: {task}\nPlan: {plan}[/dim]".format(
                task=current_state.get('task', 'N/A'),
                plan='\n'.join(current_state.get('plan', []))
            ), title="Last Session"))
            resume = Prompt.ask("이전 작업을 이어서 진행하시겠습니까?", choices=["yes", "no"], default="yes")
            if resume.lower() == "no":
                current_state = None
        if not current_state:
            current_state = {
                "messages" : [SystemMessage(content=command_prompt)],
                "action_history": [],
                "extracted_data": {},
                "current_plan_step": 0,
                "plan": [],
            }

        while True:
            console.print(Panel("[bold]Please enter your request (type 'exit' or 'quit' to stop):[/bold]", border_style="blue"))
            user_input = Prompt.ask("[bold]You[/bold]")
            if user_input.lower() in ["exit", "quit"]:
                break

            if user_input:
                if not current_state.get("task"):
                    current_state["task"] = user_input
                
                current_state['messages'].append(
                    HumanMessage(content = user_input)
                )

                try:
                    async for event in app.astream(current_state, {"recursion_limit": 20}):
                        # 각 노드에서 반환된 업데이트를 current_state에 병합
                        for _, node_update in event.items():
                            if not isinstance(node_update, dict):
                                continue
                            for field_name, field_value in node_update.items():
                                if field_name == "messages" and isinstance(field_value, list):
                                    current_state.setdefault("messages", []).extend(field_value)
                                elif field_name == "action_history" and isinstance(field_value, list):
                                    current_state.setdefault("action_history", []).extend(field_value)
                                elif field_name == "extracted_data" and isinstance(field_value, dict):
                                    current_state.setdefault("extracted_data", {}).update(field_value)
                                else:
                                    current_state[field_name] = field_value

                        # 이벤트 병합 후 저장
                        save_agent_state(current_state)

                    # 마지막 응답 선택: 가장 최근 AI 메시지 우선, 없으면 마지막 메시지
                    final_msg = None
                    for msg in reversed(current_state["messages"]):
                        if isinstance(msg, AIMessage):
                            final_msg = msg
                            break
                    if final_msg is None and current_state["messages"]:
                        final_msg = current_state["messages"][-1]

                    if final_msg and getattr(final_msg, "content", None):
                        console.print(Panel(f"[bold green]🤖 Agent:[/bold green]\n{final_msg.content}", border_style="green"))
                
                except GraphRecursionError as e:
                    console.print(
                        Panel(f"[bold red]❌ I'm sorry. The task was interrupted because the recursion limit was reached.[/bold red]", border_style="red")
                    )
                    # Generate a summary of the current state for the user
                    summary = summarize_agent_state(current_state)
                    
                    final_response = model.invoke(
                        [
                            SystemMessage(
                                content=f"""
                                You are an AI assistant specialized in completing user tasks. A technical error has interrupted the current task, and you cannot proceed. Your new objective is to inform the user about this issue in a clear, helpful, and transparent manner.

                                    Your response MUST follow these steps:
                                    1.  **Acknowledge the Interruption:** Start by politely informing the user that the task could not be fully completed due to a technical issue. Be friendly and apologetic.
                                    2.  **Report Progress:** Based on the provided summary of the agent's state, detail what has been successfully accomplished so far. Specifically, extract and highlight any final, actionable information that was found. For example, if the task was to find a movie to book, clearly state the movie title, showtime, and any other relevant details that were successfully retrieved.
                                    3.  **Explain the Point of Failure:** Briefly and in plain language, explain where the process was interrupted (e.g., "영화를 찾는 데는 성공했지만, 예약 과정에서 문제가 발생했습니다" or "원하시는 정보를 찾았으나, 다음 단계로 진행할 수 없었습니다").
                                    4.  **Offer Next Steps:** End by apologizing for the inconvenience and offering to assist with a new task. Your goal is to provide a complete, standalone message that is helpful despite the failure.
                                    5.  **Maintain a Friendly and Non-technical Tone:** Avoid jargon. Your entire response should be a single, coherent message to the user, not a list of raw data.
                                    6.  **You Must answer in Korean.**
                               """
                            ), 
                            HumanMessage(
                                content=summary
                            )
                        ]
                    )
                    # Output the summary to the user
                    console.print(Panel(f"[bold green]🤖 Agent:[/bold green]\n{final_response.content}", border_style="green"))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Process interrupted by user. Exiting...[/bold red]")
