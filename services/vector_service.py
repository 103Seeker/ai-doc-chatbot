import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ── ChromaDB client (persistent local storage) ──
CHROMA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)

# ── Embeddings: free, runs locally, no API key needed ──
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ── Text splitter config ──
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)


def _get_collection(doc_id: str):
    """Get or create a ChromaDB collection for a document."""
    return client.get_or_create_collection(
        name=doc_id,
        metadata={"hnsw:space": "cosine"}
    )


def store_chunks(doc_id: str, text: str) -> int:
    """
    Split text into chunks, embed them, and store in ChromaDB.
    Returns number of chunks stored.
    """
    collection = _get_collection(doc_id)

    # Clear existing chunks for this doc (re-upload case)
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Split into chunks
    chunks = splitter.split_text(text)

    if not chunks:
        return 0

    # Embed all chunks
    vectors = embeddings_model.embed_documents(chunks)

    # Store in ChromaDB
    collection.add(
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"chunk_index": i, "doc_id": doc_id} for i in range(len(chunks))]
    )

    return len(chunks)


def search_chunks(doc_id: str, query: str, top_k: int = 3) -> list[dict]:
    """
    Semantic search: find top-k chunks most relevant to the query.
    Returns list of {text, score, chunk_index}.
    """
    collection = _get_collection(doc_id)

    query_vector = embeddings_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"]
    )

    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "text": doc,
            "score": round(1 - results["distances"][0][i], 4),
            "chunk_index": results["metadatas"][0][i]["chunk_index"]
        })

    return output


def clear_store(doc_id: str):
    """Delete a document's ChromaDB collection."""
    try:
        client.delete_collection(name=doc_id)
    except Exception:
        pass