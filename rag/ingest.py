import os
import uuid
from utils.supabase_client import supabase
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ingest_document(data: dict):
    file_name = data.get("file")
    project = data.get("project")
    tags = data.get("tags", [])

    # Normally you’d parse file content here via Docupipe or unstructured.io
    text = f"Placeholder content extracted from {file_name}"

    # Generate embedding
    embedding = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    ).data[0].embedding

    # Store metadata in Supabase
    doc_id = str(uuid.uuid4())
    supabase.table("documents").insert({
        "id": doc_id,
        "title": file_name,
        "project": project,
        "source": "manual",
    }).execute()

    # Upsert to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    index.upsert([(doc_id, embedding, {"project": project, "tags": tags})])

    return {"status": "success", "doc_id": doc_id}
