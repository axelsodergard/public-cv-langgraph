import sys
import asyncio
from langchain_core.messages import HumanMessage
from utils import app, ChatState, print_ai_message
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

print("\n\n")
print_ai_message("Welcome to the AI CV generation chat. Type 'exit' to quit.")
print_ai_message("The first phase is collect info, write PROCEED to move on to create the CV\n")

# Importing agent app (assuming it's an async function handler)

# FastAPI instance
api_app = FastAPI()

# Pydantic model for API requests
class ChatRequest(BaseModel):
    user_input: str

@api_app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """API endpoint to interact with the AI chat."""
    try:
        input_dict: ChatState = {
            "messages": [HumanMessage(content=request.user_input)],
            "next_step": "call_first_model"
        }
        output = await app.ainvoke(input_dict, {"configurable": {"thread_id": "abc123"}})
        return {"response": output["messages"][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CLI mode (kept as is)
async def main():
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat.")
            sys.exit()

        input_dict: ChatState = {
            "messages": [HumanMessage(content=user_input)],
            "next_step": "call_first_model"
        }

        output = await app.ainvoke(input_dict, {"configurable": {"thread_id": "abc123"}})
        print_ai_message(output["messages"][-1].content)
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())
