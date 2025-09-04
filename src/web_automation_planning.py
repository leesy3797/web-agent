"""User Clarification and Research Brief Generation.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research.
"""

from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from src.prompts import clarification_prompt_template, automation_plan_prompt_template
from src.state_automation import AgentState, Plan, Clarification
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

load_dotenv()

console = Console()

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %d, %Y")

# ===== CONFIGURATION =====

# Initialize model
model = init_chat_model(
    model='google_genai:gemini-2.0-flash', 
    temperature=0.0
)

# ===== WORKFLOW NODES =====

def clarify_with_user(state: AgentState) -> Command[Literal["planning_automation", "clarify_with_user"]]:
    """
    Determine if the user's request contains sufficient information to automate Web Tasks.

    Routes to either research brief generation or back to itself for clarification.
    """
    console.print(Panel(f"[bold blue]🔍 Clarification Node - Processing...[/bold blue]", border_style="blue"))
    console.print(f"[dim]Current state keys: {list(state.keys())}[/dim]")
    
    structured_output_model = model.with_structured_output(Clarification)

    response = structured_output_model.invoke([
        HumanMessage(content=clarification_prompt_template.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])
    
    console.print(f"[bold]Clarification Response:[/bold] need_clarification={response.need_clarification}")
    console.print(f"[dim]Question: {response.question}[/dim]")
    console.print(f"[dim]Message: {response.message_to_user}[/dim]")

    if response.need_clarification:
        console.print(Panel("[bold red]❌ Need clarification - Ask Users for supplementary informations[/bold red]", border_style="red"))
        # We return a Command to send the message to the user and interrupt the workflow
        # The user's reply will then go back to this same node.
        console.print(Panel(f"[bold yellow]Question: {response.question}[/bold yellow]", border_style="red"))
        answer = Prompt.ask(f"[bold]You[/bold]")
        return Command(
            goto="clarify_with_user",
            update={
                "messages": [AIMessage(content=response.question), HumanMessage(content=answer)],
            }
        )
    else:
        console.print(Panel("[bold green]✅ No clarification needed - Routing to planning_automation[/bold green]", border_style="green"))
        return Command(
            goto="planning_automation", 
            update={
                "messages": [AIMessage(content=response.message_to_user)],
            }
        )

def human_approval(state: AgentState) -> Command[Literal["planning_automation", END]]:
    """
    Pause the workflow to get human approval before proceeding with the plan.
    """
    console.print(Panel(f"[bold magenta]👤 Human Approval Node - Processing...[/bold magenta]", border_style="magenta"))
    console.print(f"[dim]Current state keys: {list(state.keys())}[/dim]")
    
    plan_show = "\n".join([f"{i+1}. {plan_state}" for i, plan_state in enumerate(state["plan"])])
    console.print(Panel(f"[bold green]Here's the Plan agent has generated:\n\n {plan_show}.[/bold green]", border_style="green"))
    console.print(Panel("[bold yellow]The agent has generated a plan. Please review and approve to proceed.[/bold yellow]", border_style="yellow"))
    approval = Prompt.ask("Do you approve the plan? (yes/no)", choices=["yes", "no"], default="yes")
    
    console.print(f"[bold]User Approval:[/bold] {approval}")

    if approval.lower() == "yes":
        console.print(Panel("[bold green]✅ User approved - Routing to END[/bold green]", border_style="green"))
        return Command(
            goto='agent',
            update={
                "messages": [AIMessage(content="User approved the plan. Proceeding with execution.")],
            }
        )
    elif approval.lower() == "no":
        feedback = Prompt.ask("Please provide your feedback or specify changes needed in the plan")
        state["messages"].append(HumanMessage(content=feedback))
        console.print(Panel("[bold red]❌ User rejected - Routing back to planning_automation[/bold red]", border_style="red"))
        return Command(
            goto="planning_automation",
            update={
                "messages": [
                    AIMessage(content="User did not approve the plan."),
                    HumanMessage(content=f"User feedback for replanning: {feedback}")
                ],
            }
        )
    else:
        # console.print(Panel("[bold red]Invalid input. Do you want to Quit Agent?[/bold red]", border_style="red"))
        answer = Prompt.ask("[bold red]Do you want to Quit Agent? (yes/no)[/bold red]", choices=["yes", "no"], default="no")
        if answer.lower() == "yes":
            return Command(
                goto=END,
                update={
                    "messages": [AIMessage(content="Invalid input received. Maybe user wants to quit Agent.")],
                }
            )
        else:
            return Command(
                goto="human_approval"
            )

def planning_automation(state: AgentState):
    """
    Create a clear, structured plan that outlines the key steps the agent must take to fulfill the user's request.
    """
    console.print(Panel(f"[bold yellow]📋 Planning Automation Node - Processing...[/bold yellow]", border_style="yellow"))
    console.print(f"[dim]Current state keys: {list(state.keys())}[/dim]")
    
    structured_output_model = model.with_structured_output(Plan)

    response = structured_output_model.invoke([
        HumanMessage(content=automation_plan_prompt_template.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])
    
    console.print(f"[bold]Plan Generated:[/bold] {len(response.steps)} steps")
    for i, step in enumerate(response.steps):
        console.print(f"[dim]  {i+1}. {step}[/dim]")

    return {
        "plan": response.steps,
        "messages": [AIMessage(content="Plan generated and ready for user approval.")],
    }

# ===== GRAPH CONSTRUCTION =====

# Build the scoping workflow
automation_builder = StateGraph(AgentState)

# Add workflow nodes
automation_builder.add_node("clarify_with_user", clarify_with_user)
automation_builder.add_node("planning_automation", planning_automation)
automation_builder.add_node("human_approval", human_approval)

# Add workflow edges
automation_builder.add_edge(START, "clarify_with_user")
automation_builder.add_edge("planning_automation", "human_approval")
# automation_builder.add_edge("human_approval", "planning_automation")

# Compile the workflow
automation_planner = automation_builder.compile()