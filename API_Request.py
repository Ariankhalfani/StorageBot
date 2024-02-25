import requests

# Get the API endpoint URL from the user
url = input("Enter the API endpoint URL: ")

# Get the input text for the chatbot from the user
input_text = input("Enter your message: ")

# Construct the data payload for the request
data = {'text': input_text}

# Send the POST request to the API endpoint
response = requests.post(url, json=data)

# Print the response from the chatbot
print("Chatbot Response:", response.json()['response'])
