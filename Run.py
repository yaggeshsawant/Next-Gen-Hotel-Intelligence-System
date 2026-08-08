"""
run.py — joins hotel.py (RAG pipeline), sentiment.py (sentiment model),
feedback.py (LangGraph feedback pipeline), and index.html (UI) into one
running Flask app.

    hotel.py     -> get_chat_reply(query)                  [RAG: retrieve + Gemini, plain call]
    sentiment.py -> analyze_sentiment(text)                 [DistilBERT sentiment model]
    feedback.py  -> process_feedback(...)                   [LangGraph: analyze -> sheet -> emails]
    index.html   -> UI, calls /predict, /chat, /feedback

Only /feedback uses LangGraph (multi-step, branches on sentiment). /chat is
a plain retrieve-then-generate function call - no graph needed there.

None of hotel.py, sentiment.py, or feedback.py know about Flask. This file
is the only place that creates the Flask app, defines routes, and starts
the server, so the UI wiring lives in exactly one spot.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from hotel import generate_response
from sentiment import analyze_sentiment
from Feedback import process_feedback

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

    if not first_name or not last_name or not email or not feedback_text:
        return jsonify({'success': False, 'message': 'All fields are required.'}), 400

    try:
        result = process_feedback(first_name, last_name, email, feedback_text)
        status_code = 200 if result.get('success') else 500
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'sentiment': 'unknown',
            'confidence': 0.0
        }), 500


if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', debug=True, port=5000, use_reloader=False)
    print("Flask server has stopped.")