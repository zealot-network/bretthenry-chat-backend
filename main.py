from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from rag.query_engine import query_router
from rag.ingest import ingest_document

app = FastAPI(title="BrettAI Backend", version="1.0")

# CORS — adjust domain after Lovable/Vercel deploy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with ["https://bretthenry.chat"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}
from sqlalchemy import create_engine
import pandas as pd
import io
import os
from datetime import datetime
from supabase import create_client

# Environment variables for database and Supabase
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# SQLAlchemy engine and Supabase client
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.post("/data")
async def upload_data(file: UploadFile, table_name: str = Form("uploads_table")):
    try:
        # Load file into DataFrame
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        elif file.filename.endswith((".json", ".jsonl")):
            content = await file.read()
            df = pd.read_json(io.BytesIO(content), lines=True)
        else:
            return {"error": "Unsupported file format."}

        # Insert rows into the designated Postgres table
        df.to_sql(table_name, engine, if_exists="append", index=False)

        # Log metadata to Supabase
        metadata = {
            "filename": file.filename,
            "table_name": table_name,
            "rows_inserted": len(df),
            "uploaded_at": datetime.utcnow().isoformat()
        }
        supabase.table("data_uploads").insert(metadata).execute()

@app.get("/data/uploads")
async def get_data_uploads():
    try:
        response = supabase.table("data_uploads").select("*").order("uploaded_at", desc=True).execute()
        data = response.data if hasattr(response, "data") else response
        return {"uploads": data}
    except Exception as e:
        return {"error": str(e)}
        return {"status": "success", **metadata}
    except Exception as e:
        return {"error": str(e)}

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    question = data.get("question")
    context = data.get("context", [])
    response = query_router(question, context)
    return response

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), project: str = Form(...), tags: str = Form("")):
    tag_list = [t.strip() for t in tags.split(",") if t]
    result = ingest_document(file, project, tag_list)
    return result
