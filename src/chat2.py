from src.prompt import get_prompt
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM
import torch
import uuid
import pandas as pd
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessageChunk
from dotenv import load_dotenv
load_dotenv()

import time,os
class Chatbot:
    def __init__(self):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.llm=self.define_llm()
        self.embeddings=self.load_embeddings()
        self.client=MongoClient(os.environ.get("CONNECTION_STRING_URI"))
        self.db=self.client["sathvika"]
        self.collection=self.db["faq_vectorstore_bge"]
        self.index_name="index-2"
        self.prompt=get_prompt()
        self.retriever=self.load_retriever()
        self.session_id=str(uuid.uuid4())
        self.messages=[]
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
        print("creating chain")
        combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)
        return retrieval_chain

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
    
    def generate_response(self,query):
        full_res=""
        start=time.time()
        for chunk in self.chain.stream({"input":query}):
            print(chunk)
            yield chunk

        end=time.time()-start
        self.messages.append({"role":"assistant","content":full_res})
        self.save_responses(query,full_res,end)

    # async def generate_response(self,query):
    #     if self.chain is None:
    #         self.get_retrieval_chain()
    #     self.messages.append({"role":"user","content":query})
    #     print("Executing query:",query)
    #     async for event in self.chain.astream_events(input={"question":query}, version="v1"):
    #         start=time.time()
    #         full_response=""
    #         print("Event:",event["event"])
    #         if event["event"] == 'on_llm_stream' or event["event"]=="on_chain_stream":
    #             print(event)
    #             chunk_content = self.serialize_aimessagechunk(event["data"]["chunk"])
    #             chunk_content_html = chunk_content.replace("\n", "<br>")
    #             full_response+=chunk_content_html
    #             print(chunk_content_html)
    #             yield f"{chunk_content_html}"

    #         if event["event"]=="on_llm_end":
    #             end=time.time()-start
    #             print("appending the results and saving the responses")
    #             self.messages.append({"role":"assistant","content":full_response})
    #             self.save_responses(query,full_response,end)
        

# chat_bot=Chatbot()
# async def main(query):
#     async for chunk in chat_bot.generate_response(query):
#         print(chunk)
# import asyncio            
# if __name__=="main":
#     asyncio.run(main(query="what is hra"))
