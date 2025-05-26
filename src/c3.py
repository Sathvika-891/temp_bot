from src.chat_db import ChatDB
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage,AIMessage
from src.models import ChatbotUserInput
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM
import torch
import uuid
import pandas as pd
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.schema import Document
from langchain_core.messages import AIMessageChunk
from dotenv import load_dotenv
load_dotenv()

import time,os
class Chatbot:
    def __init__(self):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.llm=self.define_llm()
        self.embeddings=self.load_embeddings()
        print("connecting to MongoDB")
        self.client=MongoClient(os.environ.get("CONNECTION_STRING_URI"))
        self.db=self.client[os.environ.get("DB_NAME")]
        self.collection=self.db[os.environ.get("EMBEDDING_COLLECTION_NAME")]

        self.index_name="index-2"
        self.retriever=self.load_retriever()
        
        self.chat_db=ChatDB()
        self.chain=self.get_retrieval_chain()
        
    def load_embeddings(self,model_path="./models_list/bge-large"):
        print("loading embedding model")
        model_kwargs={"device":self.device}
        embeddings=HuggingFaceEmbeddings(model_name=model_path,model_kwargs=model_kwargs)
        return embeddings

    def define_llm(self,model_path="./models_list/llama_3_2_1B"):

        print("loading llm")
        model_id=model_path
        # for flan-T5
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model=AutoModelForCausalLM.from_pretrained(model_id)
        print("model is loaded")
        pipe = pipeline("text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=10000)
        hf = HuggingFacePipeline(pipeline=pipe,callbacks=[StreamingStdOutCallbackHandler()])
        return hf

    def load_documents(self,data_path="./data/new_processed_FAQ_30.csv"):
        print("loading documents")
        df=pd.read_csv(data_path)
        docs=[
        Document(
            page_content=ans,
            metadata={"source":ques},
            )for ques,ans in zip(df["Question"],df["Answer"])
        ]
        return docs

    def load_retriever(self):
        docs=self.collection.find().to_list()
        vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name=self.index_name,
                relevance_score_fn="cosine",
            )
        if len(docs)>0:
            retriever=vector_store.as_retriever()
        else:
            vector_store.create_vector_search_index(dimensions=1024)
            new_docs=self.load_documents()
            print("adding documents")
            vector_store.add_documents(new_docs)
            retriever=vector_store.as_retriever()
        return retriever

    def get_retrieval_chain(self):
        print("creating chain")
        memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=memory)
        return chain
    
    def serialize_aimessagechunk(self, chunk):
        if isinstance(chunk, AIMessageChunk):
            return chunk.content or ""
        elif isinstance(chunk, str):
            return chunk
        elif isinstance(chunk, dict):
            return chunk.get("content", "")
        return ""

    
    def generate_response(self, chat_input:ChatbotUserInput):
        query=chat_input.query
        uid=chat_input.user_id
        sid=chat_input.session_id
        messages=self.chat_db.get_messages(user_id=uid,session_id=sid)
        messages.append(HumanMessage(query))
        self.chain.memory.chat_memory.messages=messages
        full_res=""
        start=time.time()
        for chunk in self.chain.stream(query):
            chunk_text = self.serialize_aimessagechunk(chunk)
            full_res += chunk_text
            yield chunk
        end = time.time() - start
        messages.append(AIMessage(full_res))
        self.chat_db.save_messages(user_id=uid, session_id=sid, messages=messages)