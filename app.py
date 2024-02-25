# app.py

from flask import Flask, render_template, request
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_path = "t5_model_epoch_20.pt"  # Path to your pre-trained model
tokenizer_path = "t5_tokenizer"  # Path to your tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    prompt = request.form['prompt']
    input_text = "chat: " + prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1, early_stopping=True)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    app.run(debug=True)
