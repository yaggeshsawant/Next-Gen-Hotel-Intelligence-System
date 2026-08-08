import os
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TypedDict, Optional

from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
from langgraph.graph import StateGraph, END

from sentiment import analyze_sentiment

load_dotenv()

# ---------- Google Sheets ----------
SHEET_NAME = "1byCVZgTbN22czrBgE_Qh1l1_GXoVuhsl_ix8kFiSdu4"   # change to your sheet key
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
        print(f"Appended to Google Sheet: {row}")
    except Exception as e:
        print(f"Google Sheets error: {e}")


# ---------- Email ----------
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", EMAIL_USER)
HOTEL_EMAIL = os.getenv("HOTEL_EMAIL", "hotel@example.com")


def send_email(to_email, subject, body, is_html=False):
    if not EMAIL_USER or not EMAIL_PASSWORD:
        print(f"\n[DEV] Would send email to {to_email}")
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
        print(f"Failed to send email to {to_email}: {e}")
        return False


# ---------- LangGraph: feedback pipeline ----------
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

    subject = f"Negative Feedback from {full_name}"
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


def process_feedback(first_name: str, last_name: str, email: str, feedback_text: str) -> dict:
    """Runs the full feedback LangGraph (analyze -> log -> email customer ->
    email hotel) and returns a dict ready to jsonify. This is the single
    function run.py's /feedback route calls."""
    initial_state = FeedbackState(
        first_name=first_name,
        last_name=last_name,
        email=email,
        feedback_text=feedback_text,
        sentiment=None,
        confidence=None,
        error=None
    )
    final_state = feedback_graph.invoke(initial_state)

    if final_state.get("error"):
        return {
            'success': False,
            'message': f"Error: {final_state['error']}",
            'sentiment': final_state.get('sentiment', 'unknown'),
            'confidence': final_state.get('confidence', 0.0)
        }

    return {
        'success': True,
        'message': 'Thank you for your feedback!',
        'sentiment': final_state.get('sentiment', 'unknown'),
        'confidence': final_state.get('confidence', 0.0)
    }