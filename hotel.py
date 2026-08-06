import os
import pickle
import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# LangGraph
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# Your DistilBERT sentiment model
from sentiment import analyze_sentiment

load_dotenv()

# ---------- Google Sheets ----------
SHEET_NAME = "1pXDqb1DJzzOQCvNJDctJxPOq-fmsCa4ohBwxLRmX3wA"          # change to your sheet name
CREDENTIALS_FILE = "hotel_creds.json"

def append_to_sheet(first_name, last_name, email, feedback, sentiment, confidence):
    try:
        creds = Credentials.from_service_account_file(
            CREDENTIALS_FILE,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_NAME).sheet1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, first_name, last_name, email, feedback, sentiment, confidence]
        sheet.append_row(row)
        print(f"✅ Appended to Google Sheet: {row}")
    except Exception as e:
        print(f"❌ Google Sheets error: {e}")

# ---------- Email ----------
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", EMAIL_USER)
HOTEL_EMAIL = os.getenv("HOTEL_EMAIL", "hotel@example.com")

def send_email(to_email, subject, body, is_html=False):
    if not EMAIL_USER or not EMAIL_PASSWORD:
        print(f"\n📧 [DEV] Would send email to {to_email}")
        print(f"Subject: {subject}\nBody:\n{body}\n")
        return True

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = subject
        if is_html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")
        return False

# ---------- RAG Setup (unchanged) ----------
CSV_PATH = "updated_processed_hotel_reviews.csv"
CHROMA_PATH = "chromadb"
EMBED_BATCH_SIZE = 512
ADD_BATCH_SIZE = 4000
GEMINI_MODEL = "gemini-3.5-flash"
TOP_K = 5

METADATA_COLUMNS = [
    "Hotel_Name", "Hotel_Address", "Average_Score",
    "Reviewer_Score", "trip_type", "room_type", "length_of_stay",
]

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["combined_text"]).reset_index(drop=True)
for col in METADATA_COLUMNS:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(-1)
    else:
        df[col] = df[col].fillna("Unknown")

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

def retrieve_hotels(query, top_k=TOP_K):
    query_embedding = embed_model.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    return results

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

def generate_response(query, context):
    return chain.invoke({"query": query, "context": context})

# ---------- LangGraph for Chatbot ----------
class ChatState(TypedDict):
    query: str
    context: Optional[str]
    response: Optional[str]
    error: Optional[str]

def retrieve_node(state: ChatState) -> dict:
    query = state["query"]
    results = retrieve_hotels(query)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return {"error": "No hotels found.", "context": ""}
    return {"context": "\n\n".join(docs)}

def generate_node(state: ChatState) -> dict:
    if state.get("error"):
        return {"response": "I couldn't find any hotels matching that criteria."}
    try:
        response = generate_response(state["query"], state["context"])
        return {"response": response}
    except Exception as e:
        return {"response": f"Error generating response: {str(e)}"}

chat_builder = StateGraph(ChatState)
chat_builder.add_node("retrieve", retrieve_node)
chat_builder.add_node("generate", generate_node)
chat_builder.set_entry_point("retrieve")
chat_builder.add_edge("retrieve", "generate")
chat_builder.add_edge("generate", END)
chat_graph = chat_builder.compile()

# ---------- LangGraph for Feedback (sequential) ----------
class FeedbackState(TypedDict):
    first_name: str
    last_name: str
    email: str
    feedback_text: str
    sentiment: Optional[str]
    confidence: Optional[float]
    error: Optional[str]

def analyze_sentiment_node(state: FeedbackState) -> dict:
    try:
        result = analyze_sentiment(state["feedback_text"])
        return {
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        }
    except Exception as e:
        return {"error": str(e), "sentiment": "unknown", "confidence": 0.0}

def append_to_sheets_node(state: FeedbackState) -> dict:
    append_to_sheet(
        state["first_name"],
        state["last_name"],
        state["email"],
        state["feedback_text"],
        state["sentiment"],
        state["confidence"]
    )
    return {}

def send_customer_email_node(state: FeedbackState) -> dict:
    if state.get("error"):
        return {}
    first_name = state["first_name"]
    sentiment = state["sentiment"]
    feedback = state["feedback_text"]
    email = state["email"]

    if sentiment == "bad":
        subject = "We value your feedback"
        body = f"""Dear {first_name},

Thank you for sharing your experience with us. We are sorry to hear about your concerns.

Your feedback: "{feedback}"

We will review this matter and work to improve. Our team may contact you for further details.

Sincerely,
Hotel Management
"""
    else:
        subject = "Thank you for your feedback!"
        body = f"""Dear {first_name},

Thank you for your kind words!

Your feedback: "{feedback}"

We are thrilled that you enjoyed your experience. We look forward to serving you again.

Best regards,
Hotel Management
"""
    send_email(email, subject, body)
    return {}

def send_hotel_email_node(state: FeedbackState) -> dict:
    # Only send hotel email if sentiment is bad
    if state.get("error") or state.get("sentiment") != "bad":
        return {}
    first_name = state["first_name"]
    last_name = state["last_name"]
    email = state["email"]
    feedback = state["feedback_text"]
    confidence = state["confidence"]
    full_name = f"{first_name} {last_name}"

    subject = f"⚠️ Negative Feedback from {full_name}"
    body = f"""Customer: {full_name} <{email}>
Feedback: {feedback}
Sentiment: bad (confidence: {confidence:.2%})

Please address the issue and consider reaching out to the customer.
"""
    send_email(HOTEL_EMAIL, subject, body)
    return {}

feedback_builder = StateGraph(FeedbackState)
feedback_builder.add_node("analyze", analyze_sentiment_node)
feedback_builder.add_node("append_to_sheets", append_to_sheets_node)
feedback_builder.add_node("send_customer", send_customer_email_node)
feedback_builder.add_node("send_hotel", send_hotel_email_node)

feedback_builder.set_entry_point("analyze")
feedback_builder.add_edge("analyze", "append_to_sheets")
feedback_builder.add_edge("append_to_sheets", "send_customer")
feedback_builder.add_edge("send_customer", "send_hotel")
feedback_builder.add_edge("send_hotel", END)

feedback_graph = feedback_builder.compile()

# ---------- Flask app ----------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    initial_state = ChatState(query=user_message, context=None, response=None, error=None)
    try:
        final_state = chat_graph.invoke(initial_state)
        if final_state.get("error"):
            return jsonify({'reply': final_state["error"]})
        return jsonify({'reply': final_state.get("response", "No response generated.")})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()
    email = data.get('email', '').strip()
    feedback_text = data.get('feedback', '').strip()

    if not first_name or not last_name or not email or not feedback_text:
        return jsonify({'success': False, 'message': 'All fields are required.'}), 400

    initial_state = FeedbackState(
        first_name=first_name,
        last_name=last_name,
        email=email,
        feedback_text=feedback_text,
        sentiment=None,
        confidence=None,
        error=None
    )
    try:
        final_state = feedback_graph.invoke(initial_state)
        if final_state.get("error"):
            return jsonify({
                'success': False,
                'message': f"Error: {final_state['error']}",
                'sentiment': final_state.get('sentiment', 'unknown'),
                'confidence': final_state.get('confidence', 0.0)
            }), 500

        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback!',
            'sentiment': final_state.get('sentiment', 'unknown'),
            'confidence': final_state.get('confidence', 0.0)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'sentiment': 'unknown',
            'confidence': 0.0
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
