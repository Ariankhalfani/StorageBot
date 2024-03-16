import os
import google.generativeai as genai

# Set the environment variable for your API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCzNWn7vGs16hc1GJD2aFEzciGKfbt7pa4'

# Configure the API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the GenerativeModel
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat()

print("Welcome to StorageBot! How can I assist you today?")
while True:
    user_input = input("You: ")

    # Send user input to Gemini and retrieve response
    gemini_response = chat.send_message(user_input)
    cleaned_response_text = gemini_response.text

    # Print Gemini-Nexus's response
    print("Gemini:", cleaned_response_text)

    if user_input.lower() == 'quit' or user_input.lower() == 'exit':
        break
