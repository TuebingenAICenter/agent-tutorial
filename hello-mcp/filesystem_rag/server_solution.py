import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from chromadb import Collection, PersistentClient
from chromadb.api import ClientAPI
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context

from recursive_file_embeddings import embedding_worker


@dataclass
class AppContext():
    """Application context with typed dependencies."""
    vector_db: ClientAPI

    documents_collection: Collection
    chunks_collection: Collection
    base_directory: Path

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    base_directory = Path(args.directory)
    base_directory.mkdir(parents=True, exist_ok=True)

    client: ClientAPI = PersistentClient(path=str(base_directory.resolve() /'.chroma'))
    documents_collection = client.get_or_create_collection("documents")
    chunks_collection = client.get_or_create_collection("chunks")
    
    stop_event = asyncio.Event()
    task = asyncio.create_task(embedding_worker(base_directory, chroma_client=client, stop_event=stop_event))
    try:
        yield AppContext(
            vector_db=client,
            documents_collection=documents_collection,
            chunks_collection=chunks_collection,
            base_directory=base_directory
        )
    finally:
        stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

mcp = FastMCP("LocalFilesystemRAG",
              instructions="""
This MCP server is a document retrieval and summarization assistant that automatically syncs with a local directory. It maintains a vector database that continuously watches the configured directory, ensuring the searchable content always reflects the current state of the files. You can use it to semantically search documents, extract content chunks, read full files, and generate summaries - all backed by real-time directory synchronization.""",
              lifespan=app_lifespan)


@mcp.tool()
def search_documents(query: str, k: int = 10) -> dict:
    """Search for documents in the vector database.

    This will return the document ids/filepaths.
    Use the retrieve_chunks tool with the document_ids argument to query for content in those files."""
    ctx = get_context()
    documents_collection = ctx.request_context.lifespan_context.documents_collection
    results = documents_collection.query(query_texts=[query], n_results=k, include=['distances'])
    return results


@mcp.tool(
        title="retrieve_chunks",
        name="retrieve_chunks",
        description="""
Retrieve chunks from documents from the vectore database based on the query string.

Arguments:

- query: query string to match results in vector database against
- k: number of results. (optional, default: 15)
- document_ids: a subset of documents to query in.
    if this argument is not given, it will search in all documents in the database. (optional, default: None)

Example: 

```
retrieve_chunks(
    query="attention mechanism",
    k=10,
    document_ids=["directory/subdirectory/texx.txt",
"my_document.pdf"])
```
        """,
        )
def retrieve_chunks(query: str, k: int = 15, document_ids: Optional[List[str]] = None) -> dict[str, Any]:
    ctx = get_context()
    if document_ids:
        results = ctx.request_context.lifespan_context.chunks_collection.query(query_texts=[query], n_results=k, where={"document_id": {"$in": document_ids}})
    else:
        results = ctx.request_context.lifespan_context.chunks_collection.query(query_texts=[query], n_results=k)
    return results

@mcp.resource("data://embedded_files")
def get_directory_tree() -> str:
    """Lists the embedded files of the RAG server.
    """
    ctx = get_context()
    base_directory = ctx.request_context.lifespan_context.base_directory
    documents_collection = ctx.request_context.lifespan_context.documents_collection
    result =documents_collection.get(include=[])
    response = ""
    for file_path in result['ids']:
        path = Path(base_directory) / file_path
        response += f"- {path}\n"
    return response

@mcp.resource("file://{filepath*}")
def get_file_content(filepath: str) -> str:
    """Retrieves content at a specific path."""
    ctx = get_context()
    base_directory = ctx.request_context.lifespan_context.base_directory
    documents_collection = ctx.request_context.lifespan_context.documents_collection

    path = Path(filepath)
    if not path.is_absolute():
        path = base_directory / path
    relative_path = path.relative_to(base_directory)
    result = documents_collection.get(ids=[str(relative_path)], include=['documents'])
    if len(result['documents']) > 0:
        return result['documents'][0]
    
    return f"No such resource {relative_path} {path}"


@mcp.prompt()
def summarize(txt: str) -> str:
    """
    A summarization prompt to get well-written summaries.
    """
    prompt = f"""
You are an expert summarizer. Here is the content of the document to summarize:

{txt}

Please summarize it according to these rules:

1. Capture main ideas and key insights.
2. Provide bullet point summary.
3. Add section-level summaries if the document is long.
4. End with overall key takeaways.
"""
    return prompt

parser = argparse.ArgumentParser(description="Start MCP Server")
parser.add_argument("--directory", type=str, default="./", help="The base directory to put resources in.")
args = parser.parse_args()

if __name__ == "__main__":
    mcp.run()

