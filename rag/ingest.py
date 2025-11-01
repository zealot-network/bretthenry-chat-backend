import os
import uuid
from fastapi import UploadFile
from utils.supabase_client import supabase
from openai import OpenAI
from pinecone import Pinecone

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_text(file: UploadFile) -> str:
    """Extract text from uploaded files (PDF, DOCX, or plain text)."""
    content = file.file.read()
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        from PyPDF2 import PdfReader
        file.file.seek(0)
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    elif filename.endswith(".docx"):
        import docx
        file.file.seek(0)
        doc = docx.Document(file.file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = content.decode("utf-8")
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    words = text.split()
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + chunk_size])


def ingest_document(file: UploadFile, project: str, tags: list):
    text = extract_text(file)
    chunks = list(chunk_text(text))
    embeddings = []
    for chunk in chunks:
        embedding = client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk
        ).data[0].embedding
        embeddings.append(embedding)
    doc_id = str(uuid.uuid4())
    supabase.table("documents").insert({
        "id": doc_id,
        "title": file.filename,
        "source": "upload",
        "project": project
    }).execute()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    vectors = []
    for emb in embeddings:
        chunk_id = str(uuid.uuid4())
        vectors.append((chunk_id, emb, {"project": project, "tags": tags}))
    index.upsert(vectors)
    return {"status": "success", "chunks": len(chunks), "doc_id": doc_id}
