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

# TODO: Add mcp tools, resources and prompts

parser = argparse.ArgumentParser(description="Start MCP Server")
parser.add_argument("--directory", type=str, default="./", help="The base directory to put resources in.")
args = parser.parse_args()

if __name__ == "__main__":
    mcp.run()

