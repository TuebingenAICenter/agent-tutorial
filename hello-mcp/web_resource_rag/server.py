import argparse
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, List, Literal, Optional

import numpy as np
from chromadb import PersistentClient
from chromadb.api import ClientAPI
from documents import PDFDocument, YouTubeVideo
from fastmcp import FastMCP
from fastmcp.exceptions import NotFoundError
from fastmcp.server.dependencies import get_context
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from loaders import YouTubeTranscriptLoader


@dataclass
class AppContext:
    """Application context with typed dependencies."""

    vector_db: ClientAPI
    documents_collection: Any
    chunks_collection: Any
    base_directory: Path


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    base_directory = Path(args.directory)
    client: ClientAPI = PersistentClient(path=str(base_directory / "vector_db.chroma"))
    documents_collection = client.get_or_create_collection("documents")
    chunks_collection = client.get_or_create_collection("chunks")
    try:
        base_directory.mkdir(parents=True, exist_ok=True)
        yield AppContext(
            vector_db=client,
            documents_collection=documents_collection,
            chunks_collection=chunks_collection,
            base_directory=base_directory,
        )
    finally:
        pass


mcp = FastMCP(
    "OnlineResourceRAG",
    instructions="""
This MCP server provides a document management system that allows LLMs to store, search, and retrieve research materials from a vector database. It handles PDFs and YouTube videos by extracting their content, splitting them into chunks, and storing both the full documents and their semantic chunks. The server enables semantic search across documents and their chunks, document listing, and deletion capabilities.

The system is designed for research workflows where an LLM needs to:
- Ingest external documents (PDFs, YouTube transcripts) for later reference
- Search through stored research materials using semantic queries
- Retrieve specific content chunks from relevant documents
- Manage the document collection by listing or removing items

An LLM should use this server when it needs to work with external research materials, maintain a persistent knowledge base across sessions, or perform semantic search across document collections.

              """,
    lifespan=app_lifespan,
)


@mcp.tool()
def add_document(url: str, doc_type: Literal["pdf", "youtube"]) -> str:
    """Adds a PDF or YouTube video document to the vector database by:
    1. Loading the document content using appropriate loader
    2. Splitting into chunks with text splitter
    3. Adding chunks to chunks collection with document ID metadata
    4. Creating document embedding by averaging chunk embeddings
    5. Adding document to documents collection with metadata and full text
    Returns confirmation message with document ID and chunk count.
    """
    ctx = get_context()
    chunks_collection = ctx.request_context.lifespan_context.chunks_collection
    documents_collection = ctx.request_context.lifespan_context.documents_collection
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_id = url

    if doc_type == "pdf":
        loader = PyPDFLoader(url)
        documents = loader.load()
        document = PDFDocument(
            source_url=url, title=documents[0].metadata.get("title", None)
        )
    elif doc_type == "youtube":
        loader = YouTubeTranscriptLoader(url)
        documents = loader.load()
        document = YouTubeVideo(title=documents[0].metadata.get("title", None))

    # Insert chunks in vector DB
    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["document_id"] = doc_id
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    chunks_collection.add(
        ids=chunk_ids,
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )

    # Insert document in vector DB
    results = chunks_collection.get(ids=chunk_ids, include=["embeddings"])
    embeddings = results["embeddings"]
    document_embedding = np.average(embeddings, axis=0)
    document_full_text = " ".join([page.page_content for page in documents])
    documents_collection.add(
        ids=[doc_id],
        embeddings=[document_embedding],
        documents=[document_full_text],
        metadatas=[document.model_dump(mode="json", exclude_none=True)],
    )
    return f"Added document {doc_id} and {len(chunks)} chunks to the vector database."


@mcp.tool()
def search_documents(query: str, k: int = 10) -> dict:
    """Searches documents in the vector database using a query string.

    Note: This returns only document IDs and metadata. Use retrieve_chunks with the
    returned document_ids to fetch actual document content.

    Args:
        query: Search text to match against document contents.
        k: Number of top results to return (default 10).

    Returns:
        A dictionary containing document IDs and metadata for the top matches.
    """
    ctx = get_context()
    results = ctx.request_context.lifespan_context.documents_collection.query(
        query_texts=[query], n_results=k, include=["distances"]
    )
    return results


@mcp.tool()
def retrieve_chunks(
    query: str, k: int = 15, document_ids: Optional[List[str]] = None
) -> dict[str, Any]:
    """
    Retrieve semantically similar text chunks from a vector database based on a query.

    This tool performs a semantic search against a collection of text chunks stored in a vector database.
    It returns the top-k most relevant chunks that match the query, optionally filtered by specific document IDs.

    Parameters:
        query (str): The search query string to find similar text chunks.
        k (int, optional): Number of top results to return. Defaults to 15.
        document_ids (Optional[List[str]], optional): List of document IDs to filter the search.
            If provided, only chunks from these documents will be considered. If None, searches all documents.

    Returns:
        dict[str, Any]: A dictionary containing the search results with the following structure:
            - 'ids': List of chunk IDs for the retrieved results
            - 'documents': List of text content for the retrieved chunks
            - 'metadatas': List of metadata dictionaries for each retrieved chunk
            - 'distances': List of similarity scores/distances for each result
    """
    ctx = get_context()
    if document_ids:
        results = ctx.request_context.lifespan_context.chunks_collection.query(
            query_texts=[query],
            n_results=k,
            where={"document_id": {"$in": document_ids}},
        )
    else:
        results = ctx.request_context.lifespan_context.chunks_collection.query(
            query_texts=[query], n_results=k
        )
    return results


@mcp.tool()
def delete_documents(doc_keys: List[str]) -> str:
    """Delete documents and their associated chunks by document keys/IDs.

    This function removes documents from the document collection and also cleans up
    any associated chunks that reference those documents via document_id.

    Args:
        doc_keys: A list of document identifiers/keys to delete

    Returns:
        A confirmation message indicating which documents were deleted
    """
    ctx = get_context()
    doc_collection = ctx.request_context.lifespan_context.documents_collection
    chunk_collection = ctx.request_context.lifespan_context.chunks_collection

    for doc_key in doc_keys:
        doc_collection.delete(ids=[doc_key])
        chunk_collection.delete(where={"document_id": doc_key})

    return f"Deleted documents {doc_keys}"


@mcp.tool()
def list_documents() -> list[dict[str, Any]]:
    """List all documents in the documents collection with their titles and sources.

    This tool provides access to the document store, returning a list of documents
    where each document is represented as a dictionary containing:
    - 'title': The document title (if available in metadata)
    - 'source': The unique identifier for the document

    Returns:
        A list of dictionaries representing documents with their metadata
    """
    ctx = get_context()
    documents_collection = ctx.request_context.lifespan_context.documents_collection
    all_documents = documents_collection.get(include=["metadatas"])
    documents = [
        {"title": metadata.get("title", None), "source": doc_id}
        for doc_id, metadata in zip(all_documents["ids"], all_documents["metadatas"])
    ]
    return documents


@mcp.tool()
def get_full_text(document_url: str) -> str:
    ctx = get_context()
    documents_collection = ctx.request_context.lifespan_context.documents_collection
    documents = documents_collection.get(
        ids=[document_url], include=["metadatas", "documents"]
    )
    if len(documents["ids"]) > 0:
        return documents["documents"][0]
    else:
        raise NotFoundError(f"File {document_url} not found.")


@mcp.resource("data://all_resources")
def list_all_resources() -> str:
    ctx = get_context()
    documents_collection = ctx.request_context.lifespan_context.documents_collection
    all_documents = documents_collection.get(include=["metadatas"])
    response = ""
    for doc_id, metadata in zip(all_documents["ids"], all_documents["metadatas"]):
        title = metadata.get("title", "N/A")
        response += f"- <{doc_id}>: {title}\n"
    return response


@mcp.resource("https://{path*}")
def get_resource(path: str) -> str:
    ctx = get_context()
    documents_collection = ctx.request_context.lifespan_context.documents_collection
    url = f"https://{path}"
    documents = documents_collection.get(ids=[url], include=["metadatas", "documents"])
    if len(documents["ids"]) > 0:
        return documents["documents"][0]
    else:
        raise NotFoundError(f"File {url} not found.")


@mcp.prompt()
def summarize(url: str) -> str:
    """
    A summarization prompt to get well-written summaries.
    """
    prompt = f"""
You are an expert summarizer. Here is the content of the document to summarize:

Load the text of the following document into context, by calling `get_full_text({url})`.
If the document doesn't exist, try to fetch it first with `add({url})`, then try again.

With the full text in the context summarize the document.

Please summarize it according to these rules:

1. Capture main ideas and key insights.
2. Provide bullet point summary.
3. Add section-level summaries if the document is long.
4. End with overall key takeaways.
"""
    return prompt


@mcp.prompt()
def generate_transcript(research_topic: str, expert_persona: str = "") -> str:
    """Generate a podcast interview transcript based on a research topic."""
    if expert_persona:
        expert_persona = f"\nThe expert has the following characteristics and personality: {expert_persona}.\n"

    return f"""You are tasked with generating a short interesting transcript of a podcast interview between an interviewer and an expert on a specific topic.
    
    <Task>
    Your task is to create a transcript of a podcast interview that is engaging, informative, and aligned with the specified research topic and goals.    
    {expert_persona}

    </Task>

    <Instructions>
    1. Analyze the input and check with the vector database for relevant information to understand the topic deeply (Use the retrieve_chunks and/or search_documents tools as needed)

    2. Based on the retrived info, if any, generate atleast 3 essential questions that, when answered, capture the main points and core meaning of the input.

    3. When formulating your questions: a. Address the central theme (or themes if there are many) or argument (or arguments if many). b. Identify key supporting ideas c. Highlight important facts or evidence d. Reveal the author's purpose or perspective e. Explore any significant implications or conclusions.

    4. Answer all of your generated questions one-by-one in detail, as an expert of the topic would.
    
    </Instructions>

    **Important reminders**:
    - Stay in character as an interviewer and expert throughout the transcript.
    - Ensure the transcript is coherent and flows naturally. So, instead of just listing questions and answers, weave them into a conversational format.
    - Make the transcript engaging and informative.
    - Avoid generic or vague statements; be specific and detailed.
    - Ensure the content is accurate and factually correct.
    - Use a conversational tone suitable for a podcast audience.
    - Keep the transcript concise, ideally between 500 to 1000 words.

    Here is your topic of research: {research_topic}
    """


@mcp.prompt()
def generate_research_questions(research_topic: str, goals: str | None = None) -> str:
    listed_goals = ""
    if goals:
        listed_goals += "\n- " + "\n- ".join(goals.split(","))

    return f"""You are an analyst tasked with interviewing an expert to learn about a specific topic. 

    Your goal is boil down to interesting and specific insights related to your topic.

    1. Do not assume, do not make up facts. Use the available tools ('search_documents' and 'retrieve_chunks') to find relevant information about the topic, if any.

    2. Based on the information you find and the given <Research Topic> and <Goals>, generate a list of 3-5 insightful and specific questions that will challenge the <Research Topic>, shed new light on the missing pieces of information. 
            
    2. Specific: Insights that avoid generalities and include specific examples from the available sources.

    Here is the ressearch topic: 
    <Research Topic>        
    {research_topic}
    </Research Topic>        
    
    These are your goals or areas of focus:
    <Goals>
    {listed_goals}
    </Goals>
    
    **Important reminders**:
    - Come up with at most 5 questions.
    - Ensure the questions are open-ended and thought-provoking.
    - Avoid yes/no questions.
    - Make sure the questions are relevant to the <Research Topic> and <Goals>.
    - Quote any sources you used to come up with the questions.
    """


parser = argparse.ArgumentParser(description="Start MCP Server")
parser.add_argument(
    "--directory",
    type=str,
    default="./research_mcp_server_directory",
    help="The base directory to put resources in.",
)
args = parser.parse_args()

if __name__ == "__main__":
    mcp.run()
