import json
from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from services.nlp_service import predict_question

app = Flask(__name__)


@app.route('/ask', methods=['POST'])
def ask():

    # Receive a question on JSON format
    data = request.get_json()  
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Predict answer
    question_id, answer = predict_question(question)

    return jsonify({'question_id': question_id, 'question': question, 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
