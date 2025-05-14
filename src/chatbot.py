from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import uuid
import pandas as pd
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessageChunk
from src.prompt import get_prompt
from dotenv import load_dotenv
load_dotenv()

import time,os
class Chatbot:
    def __init__(self):
        
        self.llm=None
        self.retriever=None
        self.chain=None
        self.embeddings=None
        self.session_id=str(uuid.uuid4())
        self.messages=[]
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.client=MongoClient(os.environ.get("CONNECTION_STRING_URI"))
        self.db=self.client["sathvika"]
        self.collection=self.db["faq_vectorstore_bge"]
        self.index_name="index-2"
        self.prompt=get_prompt()

    def load_embeddings(self,model_path="./models_list/bge-large"):
        if self.embeddings is None:
            print("loading embedding model")
            model_kwargs={"device":self.device}
            embeddings=HuggingFaceEmbeddings(model_name=model_path,model_kwargs=model_kwargs)
            self.embeddings=embeddings

    def define_llm(self,model_path="./models_list/flan-t5-small"):

        print("loading llm")
        model_id=model_path
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=10)
        hf = HuggingFacePipeline(pipeline=pipe)
        self.llm=hf

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
        if len(docs)>0:
            vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name=self.index_name,
                relevance_score_fn="cosine",
            )
            self.retriever=vector_store.as_retriever()
        else:
            vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name=self.index_name,
                relevance_score_fn="cosine",
            )
            vector_store.create_vector_search_index(dimensions=1024)
            new_docs=self.load_documents()
            print("adding documents")
            vector_store.add_documents(new_docs)
            self.retriever=vector_store.as_retriever()

    def format_docs(self,docs):
        print("documents returned are",docs)
        return "\n\n".join(doc.page_content for doc in docs[:3])

    def format_history(self):
        formatted_history=""
        for message in self.messages:
            if message["role"]=="user":
                formatted_history+="Human :".format(message["content"])
            elif message["role"]=="assistant":
                formatted_history+=f"Assistant:".format(message["content"])
        return formatted_history

    def get_retrieval_chain(self):
        if self.embeddings is None:
            self.load_embeddings()
        if self.llm is None:
            print("loading llm")
            self.define_llm()
        if self.retriever is None:
            print("loading vectorstore")
            self.load_retriever()
        self.chain=(
            {"context": self.retriever | self.format_docs,
            # "chat_history":lambda x:self.format_history(),
            "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    def serialize_aimessagechunk(self,chunk):
        """
        Custom serializer for AIMessageChunk objects.
        Convert the AIMessageChunk object to a serializable format.
        """
        if isinstance(chunk, AIMessageChunk):
            return chunk.content
        elif isinstance(chunk,str):
            return chunk
        else:
            raise TypeError(
                f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
            )
    def save_responses(self,query,response,response_time):
        context=self.retriever.get_relevant_documents(query)
        metadata={
            "token_count":len(response),
            "response_time":round(response_time,2),
            "relevant_docs":context
        }
        new_df=pd.DataFrame(
            {
                "session_id":[self.session_id],
                "query":[query],
                "response":[response],
                "metadata":[metadata]
            }
        )
        if not os.path.exists("results"):
            os.mkdir("results")
        save_path="results/{}.csv".format(self.session_id)
        new_df.to_csv(save_path,index=False)
        print("result is saved too {}".format(save_path))


    async def generate_response(self,query):
        if self.chain is None:
            self.get_retrieval_chain()
        self.messages.append({"role":"user","content":query})
        print("Executing query:",query)
        async for event in self.chain.astream_events(query, version="v1"):
            start=time.time()
            full_response=""
            
            if event["event"] == 'on_llm_stream':
                chunk_content = self.serialize_aimessagechunk(event["data"]["chunk"])
                chunk_content_html = chunk_content.replace("\n", "<br>")
                full_response+=chunk_content_html
                print(chunk_content_html)
                yield f"{chunk_content_html}"
            end=time.time()-start
        self.messages.append({"role":"assistant","content":full_response})
        self.save_responses(query,full_response,end)