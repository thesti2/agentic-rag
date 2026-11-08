from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-6-luna",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


# outpout structured
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'",
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """
    You are a grader assessing the relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade is as relevant.
    Give a binary 'yes' or 'no' score to indicate whether the document is relevant to the question.    
"""

grade_prompt = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User Question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader