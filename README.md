# StorageBot
This is korika weekly mentoring project by Team UGM. we build fine-tuning T5 Model for customer service purposes.
In this model we use sample of 500 Conversation and model with 0.06T (60 Miliion Parameters)
If you want to deploy flask app.py, create a folder with templates folder in it. then put your index.html file on it. 

## Setup

To set up StorageBot, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/ariankhalfani/StorageBot.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the necessary API keys and environment variables.
      - edit .env.example
      - add your keys and `seed prompts`

4. Add your files the folder `add_your_files_here` ; supported formats : ...

5. Run the application:
   ```
   python app.py
   ```
