import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentiment import analyze_sentiment  

load_dotenv()

CSV_PATH = "updated_processed_hotel_reviews.csv"   # change if needed
CHROMA_PATH = "chromadb"
EMBED_BATCH_SIZE = 512
ADD_BATCH_SIZE = 4000
GEMINI_MODEL = "gemini-3.5-flash"   # or "gemini-1.5-flash"
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
client_db = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client_db.get_or_create_collection(name="hotels_updated")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

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

if collection.count() == 0:
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
    results = retrieve_hotels(query)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "I couldn't find any hotels matching that criteria."
    context = "\n\n".join(docs)
    return chain.invoke({"query": query, "context": context})

# ---------- Flask app ----------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')
    
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    review = data.get('review', '').strip()
    if not review:
        return jsonify({'error': 'No review text provided.'}), 400

    try:
        result = analyze_sentiment(review)
        return jsonify({
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'probabilities': result.get('probabilities', {})
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    try:
        reply = generate_response(user_message)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()
    email = data.get('email', '').strip()
    feedback_text = data.get('feedback', '').strip()

    # Simple validation
    if not first_name or not last_name or not email or not feedback_text:
        return jsonify({'success': False, 'message': 'All fields are required.'}), 400

    # For now, print to console (you can save to CSV or DB)
    print(f"Feedback from {first_name} {last_name} <{email}>: {feedback_text}")

    # Example: append to a CSV file
    # import csv
    # with open('feedback.csv', 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([first_name, last_name, email, feedback_text])

    return jsonify({'success': True, 'message': 'Thank you for your feedback!'})
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
