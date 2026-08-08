import os
# pyrefly: ignore [missing-import]
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# Setup Gemini Client
api_key = os.environ.get("GEMINI_API_KEY") or "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg"
client = genai.Client(api_key=api_key)

def load_document():
    # Look for sample_document.txt in various locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "sample_document.txt"),
        os.path.join(os.path.dirname(__file__), "..", "sample_document.txt"),
        os.path.join(os.path.dirname(__file__), "..", "llm_assignments", "sample_document.txt"),
    ]
    
    doc_path = None
    for path in possible_paths:
        if os.path.exists(path):
            doc_path = path
            break
            
    if not doc_path:
        raise FileNotFoundError("Could not find 'sample_document.txt' in any of the search locations.")
        
    print(f"[1/4] Loading document from: {os.path.abspath(doc_path)}")
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def chunk_text(text):
    # Split by double newlines and filter out empty lines or titles
    raw_chunks = text.split("\n\n")
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if chunk and not chunk.startswith("====="):
            chunks.append(chunk)
            
    print(f"      Split document into {len(chunks)} text chunks.")
    return chunks

def build_vector_index(chunks):
    print("[2/4] Generating embeddings and building FAISS index...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return embedder, index

def retrieve_context(query, chunks, embedder, index, k=2):
    print(f"[3/4] Retrieving top {k} relevant chunks for query: '{query}'...")
    query_embedding = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k=k)
    
    retrieved_chunks = []
    print("\n   --- Retrieved Context ---")
    for rank, idx in enumerate(indices[0], 1):
        if idx < len(chunks):
            chunk = chunks[idx]
            retrieved_chunks.append(chunk)
            print(f"   [{rank}] (Distance: {distances[0][rank-1]:.4f})")
            print(f"       {chunk}")
    print("   -------------------------\n")
    
    return "\n\n".join(retrieved_chunks)

def generate_answer(query, context):
    print("[4/4] Generating answer from Gemini LLM...")
    prompt = f"""
You are a knowledgeable QA assistant. Answer the user's question using ONLY the provided context.
If the answer cannot be found in the context, respond with "I cannot answer this based on the provided document."

Context:
{context}

Question:
{query}

Answer:
"""
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text.strip()

def main():
    print("=" * 60)
    print("Starting RAG-Based Question Answering System")
    print("=" * 60)
    
    try:
        text = load_document()
        chunks = chunk_text(text)
        embedder, index = build_vector_index(chunks)
        
        query = input("Ask a question about the document (or press enter for default): ")
        if not query.strip():
            query = "What features are introduced in the Q4 product release and what is the customer feedback?"
            
        context = retrieve_context(query, chunks, embedder, index, k=2)
        answer = generate_answer(query, context)
        
        print("\n" + "=" * 30 + " ANSWER " + "=" * 30)
        print(answer)
        print("=" * 68 + "\n")
        
    except Exception as e:
        print(f"Error in RAG QA System: {e}")

if __name__ == "__main__":
    main()
