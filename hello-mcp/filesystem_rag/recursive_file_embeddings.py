#!/usr/bin/env python3
"""
Robust cross-platform directory watcher using Watchdog library.
Monitors a directory recursively for file changes and triggers custom functions.
"""

import asyncio
import itertools
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
from chromadb.api import ClientAPI as ClientAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer


class FileChangeHandler(FileSystemEventHandler):
    """Custom handler for file system events."""
    
    def __init__(self, callback=None, file_extensions=None, ignore_temp=True):
        """
        Initialize the handler.
        
        Args:
            callback: Function to call when files change (receives file path)
            file_extensions: Set of extensions to monitor (e.g., {'.py', '.txt'})
            ignore_temp: Whether to ignore temporary files
        """
        super().__init__()
        self.callback = callback or self.default_callback
        self.file_extensions = file_extensions
        self.ignore_temp = ignore_temp
        self.store = {'created': set(), 'modified': set(), 'deleted': set()}
        
    def should_process_file(self, file_path):
        """Determine if a file should trigger the callback."""
        path = Path(file_path)
        
        # Skip directories
        if path.is_dir():
            return False
            
        # Skip temporary files if requested
        if self.ignore_temp:
            temp_patterns = ['.tmp', '.swp', '~', '.lock', '#']
            if any(pattern in path.name for pattern in temp_patterns):
                return False
                
        # Check file extensions if specified
        if self.file_extensions:
            return path.suffix.lower() in self.file_extensions
            
        return True
    
    def on_modified(self, event):
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent) and not event.is_directory:
            if self.should_process_file(event.src_path):
                self.store['modified'].add(event.src_path)
                self.callback(event.src_path, 'modified')
    
    def on_created(self, event):
        """Handle file creation events."""
        if isinstance(event, FileCreatedEvent) and not event.is_directory:
            if self.should_process_file(event.src_path):
                self.store['created'].add(event.src_path)
                self.callback(event.src_path, 'created')

    def on_deleted(self, event):
        if isinstance(event, FileDeletedEvent) and not event.is_directory:
            if self.should_process_file(event.src_path) and not event.is_directory:
                if event.src_path in self.store['created']:
                    self.store['created'].remove(event.src_path)
                    self.store['modified'].discard(event.src_path)
                else:
                    self.store['deleted'].add(event.src_path)
                self.callback(event.src_path, 'deleted')

    def clear_store(self):
        self.store = {'created': set(), 'modified': set(), 'deleted': set()}
    
    def default_callback(self, file_path, event_type):
        """Default callback function."""
        print(f"File {event_type}: {file_path}")


class RecursiveFileEmbedder:
    """Main directory watcher class."""
    
    def __init__(self, directory, chroma_client: ClientAPI, file_extensions={'.md', '.txt', '.pdf'}, ignore_temp=True, callback=None):
        """
        Initialize the directory watcher.
        
        Args:
            directory: Path to directory to watch
            file_extensions: Set of file extensions to monitor
            ignore_temp: Whether to ignore temporary files
            callback: Function to call on file changes
        """
        self.directory = Path(directory).resolve()
        self.observer = Observer()
        self.handler = FileChangeHandler(callback, file_extensions, ignore_temp)
        self.chroma_client: ClientAPI = chroma_client 
        self.documents_collection = self.chroma_client.get_or_create_collection("documents")
        self.chunks_collection = self.chroma_client.get_or_create_collection("chunks")
        
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.directory}")
            
    def start(self):
        """Start watching the directory."""
        print(f"Starting to watch directory: {self.directory}")
        self.sync_directory_modifications()
        self.observer.schedule(self.handler, str(self.directory), recursive=True)
        self.observer.start()
        
    def stop(self):
        """Stop watching the directory."""
        print("Stopping directory watcher...")
        self.observer.stop()
        self.observer.join()

    def sync_directory_modifications(self):
        files = set(itertools.chain(
            self.directory.glob("**/*.pdf"), 
            self.directory.glob("**/*.md"), 
            self.directory.glob("**/*.txt")
        ))

        chroma_documents = self.documents_collection.get(include=['metadatas'])
        chroma_files = {self.directory / doc_id for doc_id in chroma_documents['ids']}
        new_files = files.difference(chroma_files)
        deleted_files = chroma_files.difference(files)

        files_last_modified = {file: datetime.fromtimestamp(file.stat().st_mtime) for file in files}
        chroma_documents_last_modified = {self.directory / doc_id: datetime.fromtimestamp(metadata['last_modified']) for doc_id, metadata in zip(chroma_documents['ids'], chroma_documents['metadatas'])}

        modified_files = set()
        for file in files:
            if file in files_last_modified.keys() and file in chroma_documents_last_modified.keys():
                if files_last_modified[file] > chroma_documents_last_modified[file]:
                    modified_files.add(file)
        
        
        for file in new_files.union(modified_files):
            print(f"Inserting new file to chroma: {file}")
            self.upsert_file(file)

        for file in deleted_files:
            print(f"Deleting file from chroma: {file}")
            self.delete_file(file)

    def upsert_file(self, file_path: Path):
        self.delete_file(file_path=file_path)
        relative_file_path = file_path.relative_to(self.directory)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_type = file_path.suffix[1:]

        if doc_type == "pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif doc_type in ('md', 'txt') :
            loader = TextLoader(file_path)
            documents = loader.load()
        else:
            raise RuntimeError(f"No Loader for file {file_path}.")

        now = datetime.now().timestamp()
        # Insert chunks in vector DB
        chunks = text_splitter.split_documents(documents)
        if len(chunks) == 0:
            chunks = documents
        for chunk in chunks:
            chunk.metadata["document"] = str(relative_file_path)
            chunk.metadata["last_modified"] = now
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        self.chunks_collection.add(
            ids=chunk_ids,
            documents=[chunk.page_content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

        # Insert document in vector DB
        results = self.chunks_collection.get(ids=chunk_ids, include=['embeddings'])
        embeddings = results['embeddings']
        document_embedding = np.average(embeddings, axis=0)
        document_full_text = " ".join([page.page_content for page in documents])
        self.documents_collection.add(
            ids=[str(relative_file_path)],
            embeddings=[document_embedding],
            documents=[document_full_text],
            metadatas=[{'last_modified': now}]
        )
        # f"Added document {relative_file_path} and {len(chunks)} chunks to the vector database."

    def delete_file(self, file_path: Path):
        relative_file_path = str(file_path.relative_to(self.directory))
        self.chunks_collection.delete(where={"document": relative_file_path})
        self.documents_collection.delete(ids=[relative_file_path])

    def sync(self):
        store_snapshot: dict[str, set] = deepcopy(self.handler.store)
        self.handler.clear_store()
        for file_path in store_snapshot['created'].union(store_snapshot['modified']):
            file_path = Path(file_path)
            print(f'c/m {file_path}')
            doc_type = file_path.suffix
            assert doc_type in self.handler.file_extensions
            self.upsert_file(file_path)
        for file_path in store_snapshot['deleted']:
            file_path = Path(file_path)
            print(f'd {file_path}')
            self.delete_file(file_path)
        

async def embedding_worker(directory, chroma_client, stop_event):
    watcher = RecursiveFileEmbedder(directory, chroma_client=chroma_client)
    watcher.start()
    try:
        while not stop_event.is_set():
            watcher.sync()
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        watcher.stop()

