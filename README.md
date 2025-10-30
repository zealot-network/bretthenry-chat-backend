# 🌭 BrettAI Full Implementation Plan (Ordered)

---

## ☁️⃣**PHASE 1 — FOUNDATION**

### **1. Create GitHub Repositories**

* **`bretthenry-chat-frontend`** → Lovable.dev project export or design scaffold.
* **`bretthenry-chat-backend`** → FastAPI or Next.js server (we’ll use FastAPI for LlamaIndex).

➡️ *Purpose:* separate codebases for clean deployment and version control.

---

### **2. Set Up Railway Account**

* Go to [railway.app](https://railway.app/).
* Create a new project → Connect it to your GitHub `bretthenry-chat-backend` repo.
* Add **Post-deployment web service** (FastAPI runtime).
* Enable build logs and automatic deploys from `main` branch.

✅ *Railway now becomes your compute environment.*

---

### **3. Provision Third-Party Services**

| Service           | Role                                | Setup                                                         |
| ----------------- | ----------------------------------- | ------------------------------------------------------------- |
| **Supabase**      | Structured DB + auth + file storage | Create new project, grab `SUPABASE_URL` and `SUPABASE_KEY`    |
| **Pinecone**      | Vector embeddings (RAG memory)      | Create index: `bretthenry-chat` with 1536 or 3072 dims        |
| **OpenAI**        | GPT-4 + embeddings                  | Generate `OPENAI_API_KEY`                                     |
| **Anthropic**     | Claude API                          | Generate `ANTHROPIC_API_KEY`                                  |
| **Google Gemini** | Factual reasoning                   | Enable Vertex API or Gemini API key                           |
| **Lovable.dev**   | Frontend builder                    | Build chat UI and brand site                                  |
| **Docupipe**      | Document ingestion                  | Deploy as microservice (can live inside same Railway project) |

---

## **PHASE 2 — DATA & INGESTION**

### **4. Configure Supabase Schema**

Create these tables in SQL Editor:

```sql
create table documents (
  id uuid primary key default gen_random_uuid(),
  title text,
  source text,
  project text,
  created_at timestamp default now()
);

create table chunks (
  id uuid primary key default gen_random_uuid(),
  doc_id uuid references documents(id),
  content text,
  metadata jsonb
);

create table queries (
  id uuid primary key default gen_random_uuid(),
  question text,
  model_used text,
  answer text,
  sources jsonb,
  created_at timestamp default now()
);
```

🤌 *Supabase now stores document metadata and conversation logs.*

---

### **5. Build Docupipe (Ingestion Layer)**

* Language: Python (FastAPI microservice)
* Responsibilities:

  1. Extract text from PDFs, DOCX, Notion, etc. (`unstructured.io` or `pdfplumber`)
  2. Clean + chunk (1000 tokens per chunk, 20% overlap)
  3. Add project + topic tags
  4. Write to Supabase
  5. Embed with `text-embedding-3-large`
  6. Upsert into Pinecone

```bash
POST /ingest
{
  "file": "Honest_CoFounder_Brett_Henry_Resume_2MB.pdf",
  "project": "Honest Coffee Roasters",
  "tags": ["entrepreneurship", "branding"]
}
```

✅ *Every upload = indexed knowledge for BrettAI.*

---

## **PHASE 3 — BACKEND (RAG ENGINE)**

### **6. Build FastAPI App on Railway**

**File:** `/main.py`

```python
from fastapi import FastAPI, Request
from rag.query_engine import query_router
from rag.ingest import ingest_document

app = FastAPI()

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    question = data["question"]
    context = data.get("context", [])
    return query_router(question, context)

@app.post("/ingest")
async def ingest(request: Request):
    data = await request.json()
    return ingest_document(data)
```

✅ *You now have `/query` and `/ingest` endpoints.*

---

### **7. Implement LlamaIndex RAG Logic**

**File:** `/rag/query_engine.py`

```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI, Anthropic, Gemini
import os

def get_engine(model="gpt-4-turbo"):
    vector_store = PineconeVectorStore(index_name="bretthenry-chat")
    embed = OpenAIEmbedding(model="text-embedding-3-large")
    llm_map = {
        "gpt-4": OpenAI(model="gpt-4-turbo", temperature=0.3),
        "claude": Anthropic(model="claude-3-sonnet-2024"),
        "gemini": Gemini(model="gemini-1.5-pro")
    }
    llm = llm_map.get(model, llm_map["gpt-4"])
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed)
    return VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

def query_router(question, context=[]):
    engine = get_engine()
    response = engine.query(question)
    return {
        "answer": str(response),
        "sources": [s.metadata for s in response.source_nodes]
    }
```

---

### **8. Add Environment Variables (Railway UI)**

In your Railway project → **Variables tab**:

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
SUPABASE_URL=
SUPABASE_KEY=
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
PINECONE_INDEX_NAME=bretthenry-chat
```

---

## **PHASE 4 — FRONTEND (LOVABLE)**

### **9. Create Lovable Project**

* Sign in to Lovable.dev
* Click **New Project**
* Add pages:

  * `/` → Home/Intro
  * `/chat` → BrettAI Interface
  * `/about` → Biography, projects
* Connect to Supabase Auth for private access (optional NDA: code `7905`).

---

### **10. Integrate Backend Endpoint**

In the **chat component logic**:

```js
async function handleQuery(input) {
  const response = await fetch("https://bretthenry-chat.up.railway.app/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: input })
  });
  const data = await response.json();
  displayResponse(data.answer, data.sources);
}
```

✅ *Frontend → Backend → RAG pipeline connected.*

---

### **11. Deploy Lovable Frontend**

* Publish to **Lovable’s internal CDN** for MVP.
* Later, export to GitHub → deploy to **Vercel** for more control + analytics.
* Domain:

  * `bretthenry.chat` → frontend (Lovable/Vercel)
  * `api.bretthenry.chat` → backend (Railway)

---

## **PHASE 5 — AUTOMATION & MAINTENANCE**

### **12. Scheduled Re-Embedding**

* Use Railway’s Cron Jobs or Supabase Functions:

  * Weekly re-embed any changed documents.
  * Rebuild Pinecone vectors.

### **13. Logging & Monitoring**

* Use **Railway Logs** for API activity.
* Optional: integrate **Helicone** or **LangFuse** for request tracing, latency, and cost analytics.

### **14. Version Control**

* Enable Railway → GitHub Auto-Deploy on commit.
* Use Lovable’s built-in versioning for frontend changes.
* Maintain a `CHANGELOG.md`.

### **15. Security & Auth**

* Use Supabase Auth or NDA code logic:

  ```python
  from fastapi import Header, HTTPException
  async def secure_endpoint(x_api_key: str = Header(...)):
      if x_api_key != os.getenv("NDA_ACCESS_CODE"):
          raise HTTPException(status_code=403, detail="Unauthorized")
  ```
* Use HTTPS only (`.railway.app` handles TLS automatically).
* Store all keys in environment variables (never in code).

---

## **PHASE 6 — OPTIMIZATION**

### **16. Add RouterQueryEngine**

* Split model use by intent:

  * Claude → “Explain,” “Write,” “Summarize”
  * GPT-4 → “Analyze,” “Calculate,” “Compare”
  * Gemini → “Research,” “Fact-Check”

LlamaIndex makes this easy:

```python
from llama_index.query_engine.router_query_engine import RouterQueryEngine
router = RouterQueryEngine.from_defaults(
  query_engine_tools=[
    ("gpt", gpt_engine),
    ("claude", claude_engine),
    ("gemini", gemini_engine)
  ]
)
```

---

### **17. Extend Docupipe**

* Automate file ingestion from:

  * Google Drive
  * Notion exports
  * Audio transcripts
* Integrate Whisper for voice-to-text ingestion.

### **18. Add Feedback Loop**

* Add 👍 / 👎 buttons in Lovable chat UI.
* Store feedback in Supabase (`feedback` table).
* Train a lightweight retrieval re-ranker (e.g., VoyageAI or Cohere Rerank).

### **19. Optional: Multi-Agent BrettAI**

* Build additional endpoints (FastAPI routers):

  * `/advisor` (business)
  * `/mentor` (leadership)
  * `/faith` (spiritual)
* Each uses a distinct sub-index (Pinecone namespace).

---

## **PHASE 7 — DEPLOYMENT & HANDOFF**

### **20. Final Checklist**

| Item             | Status              |
| ---------------- | ------------------- |
| Lovable Frontend | ✅ Live              |
| Railway Backend  | ✅ Running           |
| Supabase         | ✅ Connected         |
| Pinecone         | ✅ Index built       |
| Docupipe         | ✅ Automated         |
| Domains          | ✅ Configured        |
| LLM Keys         | ✅ Secured           |
| NDA Auth         | ✅ Active            |
| Analytics        | ✅ Hooked (optional) |

---

# ✅ **Final Architecture Overview**

```
[ User (Bretthenry.chat) ]
        │
        ▼
[ Lovable Frontend ] ──▶ [ Railway API (FastAPI) ]
        │                         │
        │                         ├─▶ [ LlamaIndex Retrieval ]
        │                         │       ├─▶ [ Pinecone ]
        │                         │       ├─▶ [ Supabase ]
        │                         │
        │                         ├─▶ [ LLM Router: GPT-4 / Claude / Gemini ]
        │                         │
        ▼                         ▼
  [ Chat UI ]                 [ Response + Sources ]
```
