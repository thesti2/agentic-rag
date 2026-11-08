from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

import os

from dotenv import load_dotenv

load_dotenv()

class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-6-luna",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question.
     Only judge whether the answer responds to what was asked. Do NOT judge whether it is
     factually correct, and do NOT use your own knowledge: the answer may describe events
     after your training data, and fact-checking is done by a separate grader.
     An answer that says it doesn't know, refuses, or talks about something else is a 'no'.
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader