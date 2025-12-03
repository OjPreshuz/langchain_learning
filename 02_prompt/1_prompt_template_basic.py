from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# 1. Prompt Template Basic Example using a template string
# Define a simple chat prompt template
# template = "Tell me a a joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template)

# # Format the prompt with a specific topic
# print("---Prompt Template Example---")
# prompt = prompt_template.invoke({"topic": "chickens"})
# print(prompt)

# # 2. Chat Prompt Template Example using multiple placeholders
# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} joke about a {animal}.
# Assistant: Sure! Here's a {adjective} joke about a {animal}:"""
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "gorilla"})
# print("---Chat Prompt Template Example with Multiple Placeholders---")
# print(prompt)

# 3 Chat Prompt Template with System and Human Messages using Tuple
messages = [
    ("system", "You are a teacher that helps student get better at {subject}"),
    ("human", "Explain to me {topic} in {subject}."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"subject": "Physics", "topic": "Refraction"})
print("---Chat Prompt Template Example with System and Human Messages---")
print(prompt)
