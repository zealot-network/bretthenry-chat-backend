import os
import io
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from supabase import create_client
import pandas as pd

app = FastAPI()

# --- Environment Variables ---
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Fail early if missing
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials missing")

# --- External Clients ---
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# ROOT / HEALTH
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "BrettAI backend running"}

@app.get("/health")
def health():
    return {"ok": True}

# -------------------------------------------------------------------
# /query  →  Chat / LLM endpoint
# -------------------------------------------------------------------

@app.post("/query")
async def query_backend(message: dict):
    """Handle chat messages and call your LLM pipeline."""
    text = message.get("message", "")
    if not text:
        return {"error": "No message provided"}

    # TODO: hook in your LlamaIndex → Pinecone → LLM flow
    reply = f"You said: {text}"
    return {"answer": reply}

# -------------------------------------------------------------------
# /ingest  →  Document ingestion (PDF, DOCX, TXT, MD)
# -------------------------------------------------------------------

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...),
                          source: str = Form("unknown")):
    """Store in Supabase, extract text, embed, push to Pinecone."""

    filename = file.filename
    contents = await file.read()

    # Upload raw file to Supabase Storage
    supabase.storage.from_("documents").upload(
        path=f"{datetime.utcnow().isoformat()}_{filename}",
        file=contents,
    )

    # TODO: extract text + run embeddings + insert into Pinecone

    # Log ingestion
    supabase.table("documents").insert({
        "filename": filename,
        "source": source,
        "uploaded_at": datetime.utcnow().isoformat(),
    }).execute()

    return {"status": "success", "filename": filename}

# -------------------------------------------------------------------
# /data  →  Structured data ingestion (CSV, XLSX, JSONL)
# -------------------------------------------------------------------

@app.post("/data")
async def ingest_data(file: UploadFile = File(...),
                      table_name: str = Form("uploads_table")):
    """Upload structured data → Postgres + Supabase metadata log."""

    filename = file.filename

    # Load file into DataFrame
    if filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)
    elif filename.endswith((".json", ".jsonl")):
        content = await file.read()
        df = pd.read_json(io.BytesIO(content), lines=True)
    else:
        return {"error": "Unsupported structured file type"}

    # Insert rows
    df.to_sql(table_name, engine, if_exists="append", index=False)

    # Log metadata
    supabase.table("data_uploads").insert({
        "filename": filename,
        "table_name": table_name,
        "rows_inserted": len(df),
        "uploaded_at": datetime.utcnow().isoformat(),
    }).execute()

    return {
        "status": "success",
        "filename": filename,
        "rows": len(df),
        "table": table_name
    }

# -------------------------------------------------------------------
# /config  →  Store YAML/JSON configs
# -------------------------------------------------------------------

@app.post("/config")
async def ingest_config(file: UploadFile = File(...)):
    """Upload configuration files to Supabase."""

    filename = file.filename
    contents = await file.read()

    supabase.storage.from_("configs").upload(
        path=f"{datetime.utcnow().isoformat()}_{filename}",
        file=contents,
    )

    supabase.table("configs").insert({
        "filename": filename,
        "uploaded_at": datetime.utcnow().isoformat(),
    }).execute()

    return {"status": "success", "filename": filename}
