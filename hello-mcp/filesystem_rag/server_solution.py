import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from chromadb import Collection, PersistentClient
from chromadb.api import ClientAPI
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context

from recursive_file_embeddings import embedding_worker


@dataclass
class AppContext():
    """Lifespan Application Context"""
    # Directory that is watched
    base_directory: Path
    # Chroma vector DB collections
    documents_collection: Collection
    chunks_collection: Collection


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    # Check that directory exists, else create it
    base_directory = Path(args.directory)
    base_directory.mkdir(parents=True, exist_ok=True)

    # Set up vector DB connection
    client: ClientAPI = PersistentClient(path=str(base_directory.resolve() /'.chroma'))
    documents_collection = client.get_or_create_collection("documents")
    chunks_collection = client.get_or_create_collection("chunks")
    
    # Start watchdog for automated embeddings
    stop_event = asyncio.Event()
    task = asyncio.create_task(embedding_worker(base_directory, chroma_client=client, stop_event=stop_event))
    try:
        # Yield the app context
        yield AppContext(
            documents_collection=documents_collection,
            chunks_collection=chunks_collection,
            base_directory=base_directory
        )
    finally:
        # Shutdown watchdog service for automated embeddings
        stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

# MCP Server objects
mcp = FastMCP(
        "LocalFilesystemRAG",
        instructions="""
This MCP server is a document retrieval and summarization assistant that automatically syncs with a local directory. It maintains a vector database that continuously watches the configured directory, ensuring the searchable content always reflects the current state of the files. You can use it to semantically search documents, extract content chunks, read full files, and generate summaries - all backed by real-time directory synchronization.""",
        lifespan=app_lifespan)

@mcp.tool()
def retrieve_chunks(query: str, n_results=10) -> dict[str, Any]:
    ctx = get_context()
    chunks_collection: Collection = ctx.request_context.lifespan_context.chunks_collection
    result: dict = chunks_collection.query(query_texts=[query], n_results=n_results)
    return result

@mcp.resource(uri='data://list-files')
def get_embedded_files() -> str:
    ctx = get_context()
    documents_collection = ctx.request_context.lifespan_context.documents_collection
    result = documents_collection.get(include=[])
    response = ""
    for file_path in result['ids']:
        response += f"- {file_path}\n"
    return response

@mcp.prompt()
def semantic_search(topic: str) -> str:
    prompt = f"""
Use the `retrieve_chunks` tool to search for '{topic}'. Use the result of the tool call to give me a list of documents (file path and title) covering that topic.
    """
    return prompt


# CLI argument parsing of directory
parser = argparse.ArgumentParser(description="Start MCP Server")
parser.add_argument("--directory", type=str, default="./", help="The base directory to put resources in.")
args = parser.parse_args()

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()

