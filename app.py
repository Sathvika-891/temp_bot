from fastapi import  FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from src.chatbot import Chatbot
from src.models import ChatbotUserInput
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
chat_session=Chatbot()

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

@app.get("/")
def root():
    return {"welcome":"home"}
@app.post("/chat_stream")
def chat_stream_events(chat_input: ChatbotUserInput):
    print("Received query:",chat_input.query)
    return StreamingResponse(chat_session.generate_response(chat_input), media_type="text/event-stream")

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",reload=True,port=8000)