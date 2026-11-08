from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import os
import re
import requests
load_dotenv()

# Resolve relative to this file so it works no matter which folder you run from
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma")
COLLECTION_NAME = "rag-chroma"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

rag_embedding = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-ada-002",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


def load_blog_post(url: str) -> Document:
    """Fetch a blog post and keep only its title and article body.

    Nav bar, table of contents, tag list and footer are dropped so they
    don't end up as junk chunks in the vector store.
    """
    response = requests.get(
        url,
        headers={"User-Agent": os.environ.get("USER_AGENT", "agentic-rag/0.1")},
        timeout=30,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_el = soup.find(class_="post-title")
    content_el = soup.find(class_="post-content")
    if content_el is None:
        raise ValueError(f"No .post-content found in {url}")

    for toc in content_el.find_all(class_="toc"):
        toc.decompose()

    # Break lines after block elements only, so inline links stay in their sentence
    for block in content_el.find_all(
        ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "tr"]
    ):
        block.append("\n\n")

    title = title_el.get_text(strip=True) if title_el else ""
    body = re.sub(r"\n\s*\n+", "\n\n", content_el.get_text()).strip()
    # Title goes in metadata only: as text it would become a tiny standalone chunk
    return Document(
        page_content=body,
        metadata={"source": url, "title": title},
    )


def build_vectorstore() -> Chroma:
    """Download the URLs, split them, and (re)create the Chroma DB.

    Only needs to be run once: `uv run python ingestion.py`
    """
    docs_list = [load_blog_post(url) for url in urls]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Start from an empty collection so re-running doesn't add duplicates
    existing = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=rag_embedding,
    )
    existing.delete_collection()

    return Chroma.from_documents(
        documents=doc_splits,
        collection_name=COLLECTION_NAME,
        embedding=rag_embedding,
        persist_directory=CHROMA_DIR,
    )


# Importing this module only opens the existing DB; it does not re-download anything
retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=rag_embedding,
).as_retriever()


if __name__ == "__main__":
    build_vectorstore()
