import random
import json
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import csv
import os
from datetime import datetime

# Load intents
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load("data.pth", map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Context memory for multi-turn conversation
context_memory = {}
booking_context = {}

# CSV file name
LOG_FILE = "chatbot_logs.csv"

# Ensure CSV has headers
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "user_input", "predicted_tag", "response", "correct", "feedback"])

# Save evaluation logs
def log_interaction(user_input, predicted_tag, response, correct=None, feedback=None):
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_input,
            predicted_tag,
            response,
            correct,
            feedback
        ])

def get_response(user_id, sentence):
    # Check for special commands first
    if sentence.lower().startswith("/book"):
        booking_item = sentence.split(maxsplit=1)[1] if len(sentence.split()) > 1 else "General Service"
        return f"✅ Booking confirmed for {booking_item}.", "booking"

    if sentence.lower().startswith("/recommend"):
        return "📌 Recommendation: Customers often buy our premium package with extended warranty.", "recommendation"

    if sentence.lower().startswith("/troubleshoot"):
        return "🛠️ Try restarting the device. If the issue persists, contact support.", "troubleshooting"

    # Preprocess input
    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    # Calculate confidence
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # High confidence → return random response
    if prob.item() > 0.75:  # Increased threshold for better accuracy
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                # Use context if available
                if user_id in context_memory:
                    context_response = f"Regarding your previous question about {context_memory[user_id]}, " + random.choice(intent["responses"])
                    return context_response, tag
                return random.choice(intent["responses"]), tag

    # Store context for follow-up questions
    context_memory[user_id] = sentence
    
    return "🤔 Sorry, I didn't quite understand. Could you rephrase or provide more details?", "unknown"

def chatbot():
    print("Chatbot is running! Type 'quit' to exit.")
    user_id = "user1"

    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        response, predicted_tag = get_response(user_id, sentence)
        print("Bot:", response)

        # Ask for feedback
        feedback = input("Was this response helpful? (yes/no/skip): ").lower()
        if feedback in ["yes", "no"]:
            correct = 1 if feedback == "yes" else 0
            log_interaction(sentence, predicted_tag, response, correct, feedback)
        else:
            log_interaction(sentence, predicted_tag, response)

if __name__ == "__main__":
    chatbot()