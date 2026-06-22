from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
RETRIEVER_K      = 6
LLM_MODEL        = "llama-3.3-70b-versatile"
EMBED_MODEL      = "Snowflake/snowflake-arctic-embed-l-v2.0"
ENSEMBLE_WEIGHTS = [0.5, 0.5]

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

# sessions[video_id] = { "retriever": ..., "history": [...] }
sessions: dict = {}

# ─────────────────────────────────────────────
# Shared model instances (loaded once at startup)
# ─────────────────────────────────────────────

model      = None
embeddings = None
str_parser = StrOutputParser()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, embeddings
    print("[*] Loading models...")
    model      = ChatGroq(model=LLM_MODEL, temperature=0.7)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("✅ Models ready.")
    yield
    sessions.clear()


app = FastAPI(title="YT-RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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


def build_retriever(docs: list[Document]):
    vectorstore    = FAISS.from_documents(docs, embeddings)
    faiss_ret      = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    bm25_ret       = BM25Retriever.from_documents(docs)
    bm25_ret.k     = RETRIEVER_K
    return EnsembleRetriever(
        retrievers=[bm25_ret, faiss_ret],
        weights=ENSEMBLE_WEIGHTS,
    )


def fetch_and_build_session(vid_id: str, language: str) -> None:
    transcript_list = YouTubeTranscriptApi().fetch(vid_id, languages=[language])
    transcript_text = " ".join(t.text for t in transcript_list)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks   = splitter.split_text(transcript_text)
    docs     = [Document(page_content=c) for c in chunks]

    sessions[vid_id] = {
        "retriever": build_retriever(docs),
        "history":   [],
    }

# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────

class LoadRequest(BaseModel):
    video_id: str
    language: str = "en"

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
    """
    Fetch the transcript for a YouTube video and build its RAG index.
    Call this once per video before sending chat messages.
    """
    try:
        fetch_and_build_session(req.video_id, req.language)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ready", "video_id": req.video_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Answer a question about a previously loaded video.
    Maintains per-video chat history automatically.
    """
    session = sessions.get(req.video_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Video not loaded. Call /load first.",
        )

    retriever = session["retriever"]
    history   = session["history"]

    # Step 1 — Optimise query
    optimized_query = (QUERY_OPTIMIZER_PROMPT | model | str_parser).invoke(
        {"query": req.message}
    )

    # Step 2 — Retrieve context
    context = format_docs(retriever.invoke(optimized_query))

    # Step 3 — Generate answer
    response = (FINAL_ANSWER_PROMPT | model | str_parser).invoke({
        "question":     optimized_query,
        "context":      context,
        "chat_history": format_chat_history(history),
    })

    # Step 4 — Persist history
    history.append(HumanMessage(content=req.message))
    history.append(AIMessage(content=response))

    return ChatResponse(answer=response)


@app.delete("/session/{video_id}")
async def clear_session(video_id: str):
    """Remove the in-memory index and history for a video."""
    sessions.pop(video_id, None)
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {"status": "ok"}
