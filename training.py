from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm
import csv

# Load pre-trained T5 model and tokenizer
model_name = "t5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Define the file path
file_path = "conversations.csv"

# Initialize an empty list to store the conversations
conversations = []

# Read the conversations from the CSV file
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    for row in reader:
        input_text, response_text = row
        conversations.append((input_text, response_text))
        
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress bar
for epoch in range(40):
    # Initialize tqdm for progress bar
    progress_bar = tqdm(total=len(conversations), desc=f'Epoch {epoch + 1}/{40}', position=0, leave=True)

    total_loss = 0.0
    for conversation in conversations:
        input_text = "chat: " + conversation[0]
        target_text = conversation[1]

        # Tokenize input and target text
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        target_ids = tokenizer.encode(target_text, return_tensors="pt")

        # Train the model
        model.train()
        model_inputs = {
            "input_ids": input_ids,
            "labels": target_ids
        }
        outputs = model(**model_inputs)
        loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({'Loss': total_loss / (progress_bar.n + 1)})  # Update the loss in the progress bar
        progress_bar.update(1)

    progress_bar.close()
    
# Save the model and tokenizer
model_save_path = f"t5_large_model_epoch_{epoch+1}.pt"
tokenizer_save_path = "t5_tokenizer"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)