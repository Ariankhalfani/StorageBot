import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load pre-trained model and tokenizer
model_path = r"C:\t5_model_epoch_40.pt"  # Path to your pre-trained model
tokenizer_path = r"C:\t5_tokenizer"  # Path to your tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

# Function for interactive prompt and response
def interact():
    while True:
        prompt = input("Enter your prompt (type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break

        input_text = "chat: " + prompt
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=50, num_return_sequences=1, early_stopping=True)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response == "":
            response = "I don't understand your request."
        print("Response:", response)

# Run the interactive prompt and response loop
interact()

