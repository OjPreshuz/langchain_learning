from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore.chat_message_history import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=langchain-learning-f4d8d
"""
# Load environment variables from .env file
load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "langchain-learning-f4d8d"
SESSION_ID = "user_session_1"
COLLECTION_NAME = "chat_history"

# Initialize Firestore client
print("Initializing Firestore client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore message history
print("Initializing Firestore message history...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)
print("Chat history initialized.")
print("Current Chat History:", chat_history.messages)

# Initialize the chat model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("Starting chat session. Type 'exit' to end the session.")

# Chat Loop
while True:
    user_input = input("User:").strip()
    if user_input.lower() == "exit":
        break

    # Add user message to Firestore chat history
    chat_history.add_user_message(user_input)

    # Retrieve the full chat history for context
    messages = chat_history.messages

    # Invoke the model with the chat history
    result = model.invoke(messages)
    response = result.content

    # Add AI message to Firestore chat history
    chat_history.add_ai_message(response)

    print(f"AI: {response}")
