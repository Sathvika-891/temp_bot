from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ChatbotUserInput(BaseModel):
    query: str
    user_id: str
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

