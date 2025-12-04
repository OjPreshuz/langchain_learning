from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a teacher in {subject}"),
    ("human", "Explain {topic} in {subject}")
])


def ensure_text(x):
    if isinstance(x, dict) and "content" in x:
        return x["content"]
    return str(x)


lowercase_output = RunnableLambda(lambda x: ensure_text(x).lower())

chain = prompt_template | model | StrOutputParser() | lowercase_output

result = chain.invoke({"subject": "Mathematics", "topic": "Calculus"})
print(result)
