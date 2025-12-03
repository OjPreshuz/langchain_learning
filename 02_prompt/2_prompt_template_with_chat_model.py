from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# template = "Tell me a a joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template)

# # Format the prompt with a specific topic
# print("---Prompt Template Example---")
# prompt = prompt_template.invoke({"topic": "Burna Boy"})
# print(prompt)

# response = model.invoke(prompt)
# print(f"Answer: {response.content}")

# 2. Chat Prompt and Model Template Example using multiple placeholders
# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} joke about a {animal}.
# Assistant: Sure! Here's a {adjective} joke about a {animal}:"""

# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "elephant"})

# response = model.invoke(prompt)
# print(f"Answer: {response.content}")

# 3 Chat Prompt Template with System and Human Messages using Tuple
messages = [
    ("system", "You are a teacher that helps student get better at {subject}"),
    ("human", "Explain to me {topic} in {subject}."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"subject": "Physics", "topic": "Refraction"})

response = model.invoke(prompt)
print(f"Answer: {response.content}")
