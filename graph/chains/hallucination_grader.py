from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

import os

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-6-luna",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


class GradeHallucinations(BaseModel):
    """Binary Score for hallucinations present in generated answer"""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

hallucination_prompt = ChatPromptTemplate (
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader