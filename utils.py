import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rich.console import Console
from rich.text import Text
import asyncio

console = Console()
def print_user_message(message):
    console.print(Text(f"You: {message}", style="bold green"))

def print_ai_message(message):
    console.print(Text(f"AI: {message}", style="bold green"))

def print_system_message(message):
    console.print(Text(f"SYSTEM: {message}", style="bold yellow"))


# Load environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "false"
load_dotenv()

# Initialize Azure OpenAI LLM
first_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.2,
)

# Initialize Azure OpenAI LLM
second_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0,
)


class ChatState(TypedDict):
    """Defines the state structure used in the workflow."""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Stores chat history
    next_step: str  # Determines which LLM to transition to

workflow = StateGraph(state_schema=ChatState)


async def call_first_model(state: ChatState):
    """First LLM collects user input and only switches when the user says 'ready for analysis'."""
    print_system_message("---------  this is the 1st LLM, collect info --------- ")

    first_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant that gathers structured information to help tailor a CV for the user. "
            "Follow a methodical approach to ask relevant questions without overwhelming the user. "
            "Ask in the following order:\n\n"
            "1. **Seniority Level**: Determine if the user is junior, mid-level, or senior, as this affects whether to include an education section.\n"
            "2. **General or Specific Application**: Ask if they are applying for a specific job. If so, prompt them to paste the job ad.\n"
            "3. **Professional Experiences**: Gather details about their experience in finance, including roles, responsibilities, and any notable achievements.\n"
            "4. **Educational Background** (if applicable): If the user wants to include education, collect relevant details.\n\n"
            "Additionally, ask about key accomplishments they want to highlight and tools they are proficient in.\n"
            "Ensure the conversation stays on track—if the user discusses unrelated topics, gently prompt them to refocus on their CV."
            "**YOU DO NOT OUTPUT A CV DRAFT, EVEN IF ASKED TO BY THE USER, INSTEAD ASK THE USER IF THEY WISH TO PROCEED TO THE NEXT STEP.**"
            "**YOU DO NOT OUTPUT A SUMMARY OF THE INFORMATION COLLECTED, ONLY PROVIDE SPECIFIC INFORMATION IF REQUESTED BY THE USER.**"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


    formatted_messages = first_prompt.format_messages(messages=state["messages"])
    response = await first_llm.ainvoke(formatted_messages)

    # Get the last user message
    last_user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower()
            break  

    # If user says 'ready for analysis', transition to second LLM
    if last_user_message and "proceed" in last_user_message:
        return {"messages": state["messages"] + [response], "next_step": "call_second_model"}

    return {"messages": state["messages"] + [response]}  


def route_next_step(state: ChatState):
    """Determines the next step based on user input."""
    print_system_message("---------  this is the route_next_step --------- ")
    next_step = state.get("next_step")  # Don't set a default value!

    if next_step == "call_second_model":
        return "call_second_model"  # Move to second LLM
    
    return END  # 🚨 Stops execution & waits for new input!
  # Default if something goes wrong




async def call_second_model(state: MessagesState):
    """Second LLM analyzes the collected information."""
    
    # Second LLM's prompt (Task: Analyze information)
    second_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant that generates a structured, professional CV based on the information provided in the chat history. "
            "This includes details about professional experiences, relevant educational background, and, if applicable, a job description for tailoring the CV to a specific role.\n\n"

            "**CV Structure:**\n"
            "- **Key Qualities:** A concise section listing hard skills or programs the user has demonstrated proficiency in.\n"
            "- **Narrative Paragraphs (Max 5, No Bullet Points):** Each paragraph should be structured as:\n"
            "  - A headline summarizing the main category of expertise or domain.\n"
            "  - A well-structured narrative (4-5 sentences) describing how the user has proven themselves in this field.\n"
            "- Only create as many paragraphs as warranted by the provided information.\n"
            "- The CV should be written entirely in **full paragraphs** and **should NOT use bullet points**.\n\n"

            "**Tailoring to a Job Ad:**\n"
            "- If the user is applying for a specific job and has provided a job description, align the CV content with the skills and qualifications the employer seeks.\n"
            "- If the user lacks certain desired qualities from the job ad, still construct a CV that truthfully represents their professional background and strengths.\n"
            
            "**Tonality:**\n"
            "- The writing should be professional yet approachable, avoiding pretentious language.\n"
            "- The CV must be grounded in the user’s input—do not add information that was not provided.\n\n"

            "Generate a polished and compelling CV in **full paragraph format**, effectively highlighting the user’s expertise while maintaining authenticity."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
    
    print_system_message("--------- this is the 2nd LLM, write CV --------- ")
    formatted_messages = second_prompt.format_messages(messages=state["messages"])
    response = await second_llm.ainvoke(formatted_messages)
    return {"messages": response}  # Final response to user

from langgraph.graph import START, StateGraph

# Initialize workflow graph
workflow = StateGraph(state_schema=MessagesState)

# Add LLM processing steps
workflow.add_node("call_first_model", call_first_model)
workflow.add_node("call_second_model", call_second_model)

# Define workflow transitions
workflow.add_edge(START, "call_first_model")   # User input → First LLM
workflow.add_conditional_edges(
    "call_first_model",
    route_next_step
)
workflow.add_edge("call_second_model", END)  # Second LLM → End


# Compile the workflow
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)