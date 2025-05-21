from fastapi import  FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from src.chat2 import Chatbot
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
chat_session=Chatbot()
@app.get("/")
def root():
    return {"welcome":"home"}
@app.get("/chat_stream/{message}")
def chat_stream_events(message: str):
    print("Received query:",message)
    return StreamingResponse(chat_session.generate_response(query=message), media_type="text/event-stream")

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",reload=True,port=8000)