from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a biologist with expertise in {field}"),
    ("human", "Explain the {topic} in the {field}")
])


def ensure_text(x):
    # Extracts text from LangChain outputs.
    if isinstance(x, dict) and "content" in x:
        return x["content"]
    return str(x)


uppercase_output = RunnableLambda(lambda x: ensure_text(x).upper())
count_words = RunnableLambda(
    lambda x: f"Word Count: {len(ensure_text(x).split())}\n{ensure_text(x)}")

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"field": "Anatomy", "topic": "Cell Structure"})
print(result)
