from fastapi import FastAPI, Request
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

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    question = data.get("question")
    context = data.get("context", [])
    response = query_router(question, context)
    return response

@app.post("/ingest")
async def ingest(request: Request):
    data = await request.json()
    return ingest_document(data)
