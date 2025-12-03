from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# load environment variables from .env file
load_dotenv()

# Initialize the Gemini 2.5 Flash model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Invoke the model with a sample question
response = model.invoke("Who is the president of Curacao")
print("Response from Gemini 2.5 Flash:", response)
print(response.content)


result = model.invoke('Who is Obafemi Martins')
print(result.content)
