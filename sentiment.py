# sentiment.py
import pickle
import tempfile
import os
import numpy as np
import tensorflow as tf
import keras_hub
from transformers import DistilBertTokenizer

def bytes_to_model(model_bytes, custom_objects=None):
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.keras")
        with open(model_path, "wb") as f:
            f.write(model_bytes)
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
    return model

# Load the saved bundle
with open("chatbot_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

label_enc = bundle["label_encoder"]
config = bundle["config"]

# Restore DistilBERT classifier
distilbert_classifier = bytes_to_model(
    bundle["distilbert_model_bytes"]
)

print("✅ DistilBERT model loaded successfully!")
print("Classes:", list(label_enc.classes_))

# Load tokenizer from Hugging Face
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Get max sequence length from config (fallback to 512)
max_seq_len = config.get("max_sequence_length", 512)

def analyze_sentiment(text):
    """
    Predict sentiment using DistilBERT.
    Returns:
        dict: {"sentiment": str, "confidence": float, "probabilities": dict}
    """
    # Tokenize with Hugging Face tokenizer
    encoded = tokenizer(
        text,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

    # Map to the keys expected by the model
    inputs = {
        'token_ids': encoded['input_ids'],          # shape (1, max_seq_len)
        'padding_mask': encoded['attention_mask']   # shape (1, max_seq_len)
    }

    # Predict (output is logits)
    logits = distilbert_classifier(inputs)
    
    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]  # shape (num_classes,)

    predicted_index = np.argmax(probabilities)
    sentiment = label_enc.classes_[predicted_index]
    confidence = float(probabilities[predicted_index])

    prob_dict = {
        label: float(prob)
        for label, prob in zip(label_enc.classes_, probabilities)
    }

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": prob_dict
    }
