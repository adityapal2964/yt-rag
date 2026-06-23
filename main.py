from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi

load_dotenv()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
RETRIEVER_K   = 6
LLM_MODEL     = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

QUERY_OPTIMIZER_PROMPT = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are an expert in semantic search. Rewrite the user's query so that it is "
        "optimized for retrieving relevant information from a vector database.\n"
        "Return ONLY the optimized query string. "
        "Do not include any introductory words, explanations, or quotes.\n\n"
        "Original Query: {query}\n"
        "Optimized Query:"
    ),
)

FINAL_ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "context", "chat_history"],
    template=(
        "You are an expert in processing vast amounts of text from YouTube video transcripts.\n"
        "Use the chat history for context on follow-up questions. "
        "Provide a clear, concise answer based ONLY on the provided video transcript.\n"
        'If the answer cannot be found in the context, say "I don\'t know based on this transcript."\n\n'
        "Chat History:\n{chat_history}\n\n"
        "Question: {question}\n\n"
        "Video Transcript Context:\n{context}"
    ),
)

# ─────────────────────────────────────────────
# In-memory session store
# ─────────────────────────────────────────────

sessions: dict = {}

# ─────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────

model      = None
str_parser = StrOutputParser()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("[*] Initializing models...")
    model = ChatGroq(model=LLM_MODEL, temperature=0.7)
    print("✅ Models ready.")
    yield
    sessions.clear()


app = FastAPI(title="YT-RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# BM25 Retriever
# ─────────────────────────────────────────────

class BM25Retriever:
    def __init__(self, chunks: list[str]):
        self.chunks    = chunks
        self.bm25      = BM25Okapi([c.lower().split() for c in chunks])

    def retrieve(self, query: str, k: int = RETRIEVER_K) -> str:
        scores  = self.bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return "\n\n".join(self.chunks[i] for i in top_idx)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def format_chat_history(messages: list) -> str:
    if not messages:
        return "No previous conversation."
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"AI: {msg.content}")
    return "\n".join(lines)


def build_session(transcript: str, vid_id: str) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(transcript)
    sessions[vid_id] = {
        "retriever": BM25Retriever(chunks),
        "history":   [],
    }

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class LoadRequest(BaseModel):
    video_id:   str
    transcript: str          # sent by the extension from the user's browser

class ChatRequest(BaseModel):
    video_id: str
    message:  str

class ChatResponse(BaseModel):
    answer: str

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.post("/load")
async def load_video(req: LoadRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")
    try:
        build_session(req.transcript, req.video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ready", "video_id": req.video_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session = sessions.get(req.video_id)
    if not session:
        raise HTTPException(status_code=404, detail="Video not loaded. Call /load first.")

    retriever = session["retriever"]
    history   = session["history"]

    optimized_query = (QUERY_OPTIMIZER_PROMPT | model | str_parser).invoke(
        {"query": req.message}
    )
    context  = retriever.retrieve(optimized_query)
    response = (FINAL_ANSWER_PROMPT | model | str_parser).invoke({
        "question":     optimized_query,
        "context":      context,
        "chat_history": format_chat_history(history),
    })

    history.append(HumanMessage(content=req.message))
    history.append(AIMessage(content=response))

    return ChatResponse(answer=response)


@app.delete("/session/{video_id}")
async def clear_session(video_id: str):
    sessions.pop(video_id, None)
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {"status": "ok"}