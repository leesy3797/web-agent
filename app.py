import operator
import asyncio
import uuid
from typing import TypedDict, Annotated, List
from functools import partial

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

import logging
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from dotenv import load_dotenv
from src.web_automation_planning import clarify_with_user, planning_automation, human_approval
from src.state_automation import AgentState
import json 

load_dotenv()
logging.basicConfig(level=logging.ERROR)

# --- 1. 초기 설정 ---
console = Console()

mcp_config = {
    "playwright": {
        "command": "npx",
        "args": [
            "-y",
            "@playwright/mcp@latest",
            "--user-data-dir=./browser_data",  # 👈 세션 유지
            "--save-session"                  # 👈 세션 저장
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
def think_tool(reflection: str) -> str:
    """Tool for strategically reflecting on the progress of a web automation task and deciding the next action.

    Args:
        reflection: Your detailed thoughts on the webpage state, task progress, obstacles, and the next action plan.

    Returns:
        A confirmation that the reflection has been recorded to inform the next decision.
    """
    console.print(Panel(f"[bold yellow]🤔 Agent's Reflection:[/bold yellow]\n{reflection}", border_style="yellow"))
    return f"Reflection recorded: {reflection}"


# --- 3. 툴 세팅 ---
async def setup_tools():
    client = await get_mcp_client()
    console.print(Panel("[bold yellow]Getting tools....[/bold yellow]", expand=False))
    mcp_tools = await client.get_tools()   # 동일 세션에서 툴 로드
    all_tools = mcp_tools 
    # + [think_tool]

    table = Table(title="Available Agent Tools", show_header=True, header_style="bold magenta")
    table.add_column("Tool Name", style="cyan", width=25)
    table.add_column("Description", style="white", width=80)
    for t in all_tools:
        description = t.description[:77] + "..." if len(t.description) > 80 else t.description
        table.add_row(t.name, description)
    console.print(table)
    console.print(f"[bold green]✓ Successfully retrieved {len(all_tools)} tools[/bold green]")
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

# AgentState를 JSON 파일로 저장하는 헬퍼 함수
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
            "plan": state.get("plan"),
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

# Add this function to app.py
def summarize_agent_state(state: AgentState) -> str:
    """
    Generates a user-facing summary of the agent's current state.
    """
    summary = "Here is a summary of the task's progress so far:\n\n"

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
            return self.tools_by_name[tool_name].invoke(tool_args)
        else:
            return await self.tools_by_name[tool_name].ainvoke(tool_args)
        

# --- 3. 시스템 프롬프트 정의 ---
command_prompt = """
You are an expert web automation agent. Your primary goal is to complete tasks on a webpage as requested by the user. You will use a conversational approach, providing updates and a final answer when the task is complete.
---
### <Task>
Your job is to use browser tools to navigate, interact with, and extract information from web pages to fulfill the user's request. Your actions should be strategic and efficient.
---
### <Available Tools>
You have access to a set of browser interaction tools and a reflection tool:
-   `browser_navigate`: The tool for navigating to a new URL. **Always use this as your first step** if the task requires visiting a new website.
-   `browser_snapshot`: **This is your primary observation tool.** Use it to get a textual representation of the current page's content and structure. You must use this after every significant browser action (e.g., navigation, click, form submission) to understand the new page state.
-   `browser_click`: Use to click on links, buttons, or any clickable elements identified in the snapshot.
-   `browser_type`: Use to type text into a single input field.
-   `browser_fill_form`: Use for filling multiple form fields at once.
-   `browser_select_option`: Use for selecting an option from a dropdown menu.
-   `browser_tabs`: For managing browser tabs (list, open, close, switch).
-   `browser_wait_for`: Use when you need to wait for a specific element or text to appear on the page before proceeding.
-   `browser_press_key`: For keyboard actions like pressing `Enter` or `Tab`.
-   `think_tool`: **CRITICAL: Use this after every tool action.** This is your reflection and planning tool. Analyze the outcome of your last action and determine your next best move.
---
### <Instructions>
Follow this methodical, human-like workflow:
1.  **Understand the Goal**: Read the user's request carefully. What is the final outcome they need?
2.  **Initial Action**: If the task requires visiting a new site, use `browser_navigate` first.
3.  **Observe and Plan**: After any browser action, use `browser_snapshot` to understand the page's current state. Then, use `think_tool` to analyze the snapshot, reflect on your progress, and plan the next step.
4.  **Execute**: Based on your reflection, choose the most appropriate browser tool to perform the next action (e.g., `browser_click`, `browser_type`).
5.  **Iterate**: Repeat the **Action -> Observe -> Think** cycle until the task is complete.
6.  **Final Response**: Once you have gathered all the necessary information or completed the task, provide a clear, concise answer to the user.
---
### <Hard Limits>
-   **Efficiency is Key**: Do not perform unnecessary actions. If you can complete the task with a few clicks, don't read every file or element on the page.
-   **Stop Immediately When**:
    -   You have successfully completed the user's request.
    -   You have encountered a clear obstacle and cannot proceed without user clarification.
---
### <Show Your Thinking>
Your reflections with `think_tool` should be structured and insightful:
-   What was the result of my last action?
-   What does the current page snapshot tell me?
-   Am I closer to the goal? What information is still needed?
-   What is the single best next action to take?
"""

# --- 4. LangGraph 워크플로우 정의 ---
# class AgentState(TypedDict): 
#     messages: Annotated[list[AnyMessage], operator.add]


async def agent_node(state: AgentState, model_with_tools):
    response = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


async def tool_node(state: AgentState, config: RunnableConfig, tool_executor: ToolExecutor):
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        try:
            output = await tool_executor(tool_call)
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        except Exception as e:
            console.print(Panel(f"[bold red]Error during tool execution:[/bold red]\n{e}", border_style="red"))
            tool_messages.append(ToolMessage(content=f"Error during tool execution: {e}", tool_call_id=tool_call["id"]))
            # return {"messages": tool_messages}
    return {"messages": tool_messages}


def should_continue(state: AgentState):
    if not state["messages"][-1].tool_calls:
        return "end"
    else:
        return "continue"


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
        
        workflow.add_edge(START, "clarify_with_user")
        workflow.add_edge("planning_automation", "human_approval")
        
        # workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        app = workflow.compile()

        messages = [SystemMessage(content=command_prompt)]
        console.print(Panel("[bold green]Web Automation Agent is ready.[/bold green]", title="🤖 Agent Ready"))

        while True:
            user_input = Prompt.ask("[bold]You[/bold]")
            if user_input.lower() in ["exit", "quit"]:
                break

            if user_input:
                messages.append(HumanMessage(content=user_input))
                current_state = {"messages": messages}
                try:
                    async for event in app.astream(current_state, {"recursion_limit": 20}):
                        
                        save_agent_state(current_state)

                        if "agent" in event:
                            messages.append(event["agent"]["messages"][-1])
                        elif "tools" in event:
                            messages.extend(event["tools"]["messages"])

                    final_response = messages[-1]

                    if final_response.content:
                        console.print(Panel(f"[bold green]🤖 Agent:[/bold green]\n{final_response.content}", border_style="green"))
                except GraphRecursionError as e:
                    console.print(Panel(f"[bold red]❌ I'm sorry. The task was interrupted because the recursion limit was reached.[/bold red]", border_style="red"))

                    # Generate a summary of the current state for the user
                    summary = summarize_agent_state(current_state)
                    
                    final_response = model.invoke(
                        [
                            SystemMessage(
                                content=f"""
                                The agent has been stopped due to a recursion limit error.
                                Based on the provided state summary below, you need to
                                explain the situation to the user in a friendly manner and
                                inform them that the task could not be completed.
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
