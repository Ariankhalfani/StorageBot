from flask import Flask, request, jsonify
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_path = "t5_model_epoch_20.pt"  # Path to your pre-trained model
tokenizer_path = "t5_tokenizer"  # Path to your tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

# Define API endpoint for model inference
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text from the request
        input_text = request.json['text']

        # Tokenize input text
        input_ids = tokenizer.encode("chat: " + input_text, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=50, num_return_sequences=1, early_stopping=True)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
