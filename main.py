# main.py
import os
import asyncio
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import hashlib

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
import redis

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Configuration
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama3-70b-8192"

DB_CONFIG = {
    "database": "bcp",
    "user": "bcp",
    "password": "developer@123",
    "host": "139.59.57.88",
    "port": 5400
}

REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0
}

app = FastAPI(title="Cosmas Ngeno - Real Estate Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

class ChatRequest(BaseModel):
    message: str
    session_id: str
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    customer_profile: Optional[Dict[str, Any]] = None

# Globals
vector_store = None
qa_chain = None
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'batch_size': 32}
)
redis_client = redis.Redis(**REDIS_CONFIG, decode_responses=True)
executor = ThreadPoolExecutor(max_workers=4)
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Utility functions
async def get_database_pool():
    return await asyncpg.create_pool(**DB_CONFIG)

async def fetch_property_data_async(pool):
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM pro.properties")
        data = [dict(row) for row in rows]
        return pd.DataFrame(data)

def process_documents_batch(documents: List[str], batch_size: int = 50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        chunks.extend(splitter.create_documents(batch))
    return chunks

async def create_optimized_vector_store(df: pd.DataFrame):
    global redis_client
    data_hash = hashlib.sha256(str(df.values.tobytes()).encode()).hexdigest()[:16]
    cache_key = f"vector_store_{data_hash}"

    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached.encode('latin1'))
        except: pass

    documents = [" | ".join(f"{col}: {val}" for col, val in row.items() if val) for _, row in df.iterrows()]
    chunks = await asyncio.get_event_loop().run_in_executor(executor, process_documents_batch, documents)
    vs = await asyncio.get_event_loop().run_in_executor(executor, FAISS.from_documents, chunks, embeddings_model)

    if redis_client:
        try:
            redis_client.setex(cache_key, 3600, pickle.dumps(vs).decode('latin1'))
        except: pass

    return vs

def create_optimized_qa_chain(vs):
    llm = ChatGroq(model_name=GROQ_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.7, max_tokens=512)
    prompt = PromptTemplate(
        template="""
You are Cosmas Ngeno, a friendly, professional real estate assistant. Help the customer based on the data.

Context:
{context}

Question:
{question}

Answer as Cosmas Ngeno:
""",
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

def extract_customer_preferences(query: str, profile: Dict) -> Dict:
    q = query.lower()
    locations = ["westlands", "karen", "kilimani", "runda", "ruaka"]
    types = ["apartment", "house", "villa"]

    if any(loc in q for loc in locations):
        profile['preferred_locations'] = list(set(profile.get('preferred_locations', []) + [loc.title() for loc in locations if loc in q]))
    if any(pt in q for pt in types):
        profile['property_type'] = next(pt.title() for pt in types if pt in q)
    for i in range(1, 6):
        if f"{i} bedroom" in q:
            profile['bedrooms'] = i
    return profile

async def get_conversational_response(query: str, history: List[ChatMessage]) -> str:
    context = "\n".join(f"{m.role.title()}: {m.content}" for m in history[-4:])
    return qa_chain.run(query + "\n\nContext:\n" + context)

@app.on_event("startup")
async def startup_event():
    global vector_store, qa_chain
    redis_client.ping()
    pool = await get_database_pool()
    df = await fetch_property_data_async(pool)
    vector_store = await create_optimized_vector_store(df)
    qa_chain = create_optimized_qa_chain(vector_store)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat: ChatRequest):
    try:
        profile = extract_customer_preferences(chat.message, {})
        response_text = await get_conversational_response(chat.message, chat.chat_history)
        return ChatResponse(
            response=response_text,
            session_id=chat.session_id,
            timestamp=datetime.utcnow(),
            customer_profile=profile
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
