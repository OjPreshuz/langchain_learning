from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define a chat prompt template with system and human messages
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a teacher who is an expert in {subject}"),
    ("human", "Explain {topic} in {subject} with examples.")
])

# Create the chain using Langchain Expression Language
chain = prompt_template | model | StrOutputParser()

# Run the chain
result = chain.invoke(
    {"subject": 'Physics', "topic": "Simple Harmonic Motion"})
print(result)
