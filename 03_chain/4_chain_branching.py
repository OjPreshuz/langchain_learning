from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that provides positive feedback."),
        ("human",
         "Provide positive feedback for the following response: {response}."),
    ]
)
negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides constructive criticism."),
    ("human",
     "Provide constructive criticism for the following response: {response}."),
])
escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that escalates issues to human support."),
    ("human",
     "The following response needs human attention: {response}. Please escalate it."),
])
neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides neutral feedback."),
    ("human",
     "Provide neutral feedback for the following response: {response}."),
])

classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert classifier."),
    ("human",
     "Classify the sentiment of the following response as Positive, Negative, Neutral, or Escalate: {response}."),
])

# First, compute classification alongside the original input
classified = {
    "class": classification_template | model | StrOutputParser(),
    "input": RunnablePassthrough(),
}

# Define the runnable branches, evaluating conditions against the classification
branches = RunnableBranch(
    (
        lambda x: "positive" in x["class"].lower(),  # type: ignore
        RunnableLambda(
            lambda y: y["input"]) | positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x["class"].lower(),  # type: ignore
        RunnableLambda(
            lambda y: y["input"]) | negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "escalate" in x["class"].lower(),  # type: ignore
        RunnableLambda(
            lambda y: y["input"]) | escalate_feedback_template | model | StrOutputParser()
    ),
    # default branch
    RunnableLambda(
        lambda y: y["input"]) | neutral_feedback_template | model | StrOutputParser()
)

# Create the full chain
chain = classified | branches

# Test it
review = "The product is fantastic and exceeded my expectations!"
result = chain.invoke({"response": review})
print(result)
