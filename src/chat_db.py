from pymongo import MongoClient
from datetime import datetime
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
import os

load_dotenv()

class ChatDB:
    def __init__(self):
        print("loading chatdb mongo")
        self.client = MongoClient(os.environ.get("CONNECTION_STRING_URI"))
        self.db = self.client[os.environ.get("CHAT_DB_NAME")]
        self.collection = self.db[os.environ.get("CHAT_COLLECTION_NAME")]

    def get_messages(self, user_id: str, session_id: str):
        user = self.collection.find_one({"user_id": user_id})
        if user:
            for session in user.get("sessions", []):
                if session["session_id"] == session_id:
                    stored_messages = session.get("messages", [])
                    reconstructed = []
                    for msg in stored_messages:
                        role = msg.get("role")
                        content = msg.get("content")
                        if role == "user":
                            reconstructed.append(HumanMessage(content=content))
                        elif role == "assistant":
                            reconstructed.append(AIMessage(content=content))
                        # Optionally handle other roles if used
                    return reconstructed
        return []


    def save_messages(self, user_id: str, session_id: str, messages: list[BaseMessage]):
        now = datetime.utcnow()
        formatted_messages=[]
        for msg in  messages:
            role=""
            if isinstance(msg,HumanMessage):
                role="user"
            elif isinstance(msg,AIMessage):
                role="assistant"
            formatted_messages.append(
                {"role":role,"content":msg.content}
            )

        user = self.collection.find_one({"user_id": user_id})

        if not user:
            # Create new user with initial session
            self.collection.insert_one({
                "user_id": user_id,
                "created_at": now,
                "sessions": [{
                    "session_id": session_id,
                    "created_at": now,
                    "updated_at": now,
                    "messages": formatted_messages
                }]
            })
        else:
            session_found = False
            for session in user.get("sessions", []):
                if session["session_id"] == session_id:
                    session_found = True
                    break

            if session_found:
                self.collection.update_one(
                    {"user_id": user_id, "sessions.session_id": session_id},
                    {
                        "$push": {"sessions.$.messages": {"$each": formatted_messages}},
                        "$set": {"sessions.$.updated_at": now}
                    }
                )
            else:
                self.collection.update_one(
                    {"user_id": user_id},
                    {"$push": {"sessions": {
                        "session_id": session_id,
                        "created_at": now,
                        "updated_at": now,
                        "messages": formatted_messages
                    }}}
                )
