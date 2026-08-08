import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "updated_processed_hotel_reviews.csv"   # change if needed
CHROMA_PATH = "chromadb"
EMBED_BATCH_SIZE = 512
ADD_BATCH_SIZE = 4000   # desired batch size for embeddings
MAX_CHROMA_BATCH = 166  # enforced by ChromaDB
GEMINI_MODEL = "gemini-3.5-flash"
TOP_K = 5

METADATA_COLUMNS = [
    "Hotel_Name", "Hotel_Address", "Average_Score",
    "Reviewer_Score", "trip_type", "room_type", "length_of_stay",
]

# ---------- load data ----------
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["combined_text"]).reset_index(drop=True)
for col in METADATA_COLUMNS:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(-1)
    else:
        df[col] = df[col].fillna("Unknown")

# ---------- Chroma ----------
# PersistentClient is the correct way to get on-disk persistence in
# chromadb 0.4+. (chromadb.Client(Settings(persist_directory=...)) is the
# old pre-0.4 API and no longer persists reliably on newer versions.)
client_db = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client_db.get_or_create_collection(name="hotels_updated")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Diagnostics ---
print("Persistence path:", os.path.abspath(CHROMA_PATH))
print("Collections available:", client_db.list_collections())

try:
    existing_count = collection.count()
    print("Count in hotels_updated:", existing_count)
except Exception as e:
    # A corrupted/incompatible chromadb segment (e.g. folder written by a
    # different chromadb version) throws here instead of returning 0.
    # Treat that the same as "empty" and rebuild from scratch, rather than
    # letting the whole app crash on import.
    print(f"Could not read existing collection ({e}); recreating it.")
    client_db.delete_collection("hotels_updated")
    collection = client_db.get_or_create_collection(name="hotels_updated")
    existing_count = 0


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

        # Split into sub-batches of <=166 for ChromaDB
        for sub_start in range(0, len(batch_texts), MAX_CHROMA_BATCH):
            sub_end = min(sub_start + MAX_CHROMA_BATCH, len(batch_texts))
            collection.add(
                documents=batch_texts[sub_start:sub_end],
                embeddings=batch_embeddings[sub_start:sub_end].tolist(),
                ids=ids[start + sub_start:start + sub_end],
                metadatas=metadatas[start + sub_start:start + sub_end],
            )
        print(f"Indexed {end}/{len(texts)}")


if existing_count == 0:
    build_index()


# ---------- retrieval ----------
def retrieve_hotels(query, top_k=TOP_K):
    query_embedding = embed_model.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    return results


# ---------- generation ----------
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

prompt = ChatPromptTemplate.from_template(
    """You are a travel assistant. Based on the following hotel review data,
recommend the best matching options for the user's request.

Query: {query}

Hotel data:
{context}

Only recommend hotels that actually appear in the data above.
If a hotel has no negative review mentioned, say 'Guests did not mention major complaints.'"""
)

chain = prompt | llm | StrOutputParser()


def generate_response(query):
    """Runs the full RAG pipeline: retrieve relevant reviews, then ask Gemini
    to turn them into a recommendation. This is the single function the UI
    layer (run.py) calls for the /chat route."""
    results = retrieve_hotels(query)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "I couldn't find any hotels matching that criteria."
    context = "\n\n".join(docs)
    return chain.invoke({"query": query, "context": context})