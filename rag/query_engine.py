import os
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI, Anthropic, Gemini


def get_engine(model: str = "gpt-4"):
    vector_store = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"))
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    llm_map = {
        "gpt-4": OpenAI(model="gpt-4-turbo", temperature=0.3),
        "claude": Anthropic(model="claude-3-sonnet-2024"),
        "gemini": Gemini(model="gemini-1.5-pro"),
    }
    llm = llm_map.get(model, llm_map["gpt-4"])

    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    return VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)


def query_router(question: str, context: list = []):
    engine = get_engine()
    response = engine.query(question)
    return {
        "answer": str(response),
        "sources": [s.metadata for s in response.source_nodes],
    }
