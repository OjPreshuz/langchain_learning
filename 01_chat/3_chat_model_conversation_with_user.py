from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = []

system_message = SystemMessage(
    content="You are a helpful assistant that provides concise answers.")
chat_history.append(system_message)

# Chat Loop
while True:
    user_input = input("User:").strip()
    if user_input.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=user_input))

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")

print("---Message History---")
print(chat_history)
