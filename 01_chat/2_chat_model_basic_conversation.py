from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# load environment variables from .env file
load_dotenv()

# Initialize the Gemini 2.5 Flash model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# # Create a conversation with system and human messages
# messages = [
#     SystemMessage(content='Solve the following math problem step by step'),
#     HumanMessage(content='Solve for y: 2y + 3 = 15'),
# ]

# # Invoke the model with the conversation messages
# result = model.invoke(messages)
# print(f"Answer:{result.content}")

messages_ = [
    SystemMessage(content='Solve the following math problems'),
    HumanMessage(content='What is 7 multiply 89 divided by 6?'),
    AIMessage(content='The answer is 103.8333'),
    HumanMessage(content='What is 15 percent of 2500?'),
]

result = model.invoke(messages_)
print(f"Answer:{result.content}")
