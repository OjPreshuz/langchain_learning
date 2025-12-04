from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel


load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", max_output_tokens=2048, temperature=0.5)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable assistant in {domain}"),
    ("human",
     "Provide a detailed overview of {topic} in {domain}, covering both its advantages and challenges.")
])

# Define pro analysis chain


def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst."),
        ("human", "List the pros of the following features: {features}.")
    ])
    return pros_template.format_prompt(features=features)

# Define cons analysis chain


def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst."),
        ("human", "List the cons of the following features: {features}.")
    ])
    return cons_template.format_prompt(features=features)

# Combine chains in parallel


def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Simplify branches with LCEL
pros_branch = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)
cons_branch = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch, "cons": cons_branch})
)
result = chain.invoke({"domain": "Renewable Energy", "topic": "Solar Power"})

print(result['branches']['pros'])
print(result['branches']['cons'])
