# ================================
# Hotel Review Chatbot with RAG (Gemini)
# ================================
# Preprocessing (cleaning, tag parsing, feature engineering) lives in
# preprocessing.ipynb. This script only indexes the already-processed
# CSV and handles retrieval + generation.

# pip install pandas chromadb sentence-transformers google-genai python-dotenv

import os

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = r"C:/agentic-ai/ML_project/Dataset/processed_hotel_reviews.csv"
CHROMA_PATH = r"c:/agentic-ai/chromadb"
EMBED_BATCH_SIZE = 512     # for SentenceTransformer.encode
ADD_BATCH_SIZE = 4000      # must stay under chromadb's per-add limit (5461)
GEMINI_MODEL = "gemini-3.5-flash"  # gemini-3-flash-preview is deprecated

METADATA_COLUMNS = [
    "Hotel_Name", "Hotel_Address", "Average_Score",
    "Reviewer_Score", "Trip_Type", "Guest_Type", "Is_Couple",
    "Room_Type", "Nights_Stayed",
]

# -------------------------------
# Step 1: Load preprocessed dataset
# -------------------------------
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["combined_text"]).reset_index(drop=True)

# Chroma metadata values must be str/int/float/bool/None -- Nights_Stayed can be
# NaN for rows with no "Stayed N nights" tag, so give it a safe default.
df["Nights_Stayed"] = df["Nights_Stayed"].fillna(-1).astype(int)

# -------------------------------
# Step 2: Initialize ChromaDB
# -------------------------------
client_db = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client_db.get_or_create_collection(name="hotels")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Step 3: Embed + index in batches
# (chromadb caps add() around ~5461 items per call; embedding the full
#  dataset in one shot is also a memory/latency problem)
# -------------------------------
def build_index():
    texts = df["combined_text"].tolist()
    metadatas = df[METADATA_COLUMNS].to_dict("records")
    ids = [str(i) for i in range(len(df))]

    for start in range(0, len(texts), ADD_BATCH_SIZE):
        end = min(start + ADD_BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]
        batch_embeddings = embed_model.encode(
            batch_texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        collection.add(
            documents=batch_texts,
            embeddings=batch_embeddings.tolist(),
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Indexed {end}/{len(texts)}")

# -------------------------------
# Step 4: Retrieval
# -------------------------------
def retrieve_hotels(query, top_k=3, filters=None):
    query_embedding = embed_model.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=filters,  # e.g. {"$and": [{"Is_Couple": True}, {"Average_Score": {"$gt": 7.5}}]}
    )
    return results

# -------------------------------
# Step 5: Generation (current google-genai SDK)
# -------------------------------
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_response(query, filters=None):
    results = retrieve_hotels(query, filters=filters)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "I couldn't find any hotels matching that criteria."
    context = "\n".join(docs)

    prompt = f"""You are a travel assistant. Based on the following hotel data, recommend options:
Query: {query}
Context:
{context}
If a hotel has no negative review, say 'Guests did not mention major complaints.'"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text

# -------------------------------
# Step 6: Run
# -------------------------------
if __name__ == "__main__":
    # Only build the index if the collection is empty (avoids re-embedding
    # the whole dataset on every run).
    if collection.count() == 0:
        build_index()

    user_query = "Suggest hotels in amsterdam with rating above 8 and suitable for couple"
    filters = {"$and": [{"Room Type": 'couple'}, {"Average_Score": {"$gt": 7.5}}]}
    print(generate_response(user_query, filters=None))
