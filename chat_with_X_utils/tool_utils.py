import os
import re
from typing import *

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


def _slug(text: str) -> str:
    """Convert text to lightweight slug keeping alnum, underscore, dash, pipe."""
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"[^0-9A-Za-z_\-\| ]+", "", text).replace(" ", "_")


def build_doc_key(doc_type: str, title: str, uploader: Optional[str] = None) -> str:
    """Build deterministic document key.

    Args:
        doc_type: Either "youtube" or "pdf".
        title: Title (video title or filename stem).
        uploader: Optional uploader/channel for YouTube.

    Returns:
        Deterministic slug-based key used to name all chunk IDs.
    """
    if doc_type == "youtube":
        uploader = uploader or "unknown"
        return f"{_slug(title)}|{_slug(uploader)}"
    return _slug(title)


def get_video_id(url: str) -> str:
    """Extract YouTube video ID from a variety of URL formats.

    Args:
        url: A YouTube watch/embed/short URL.

    Returns:
        11-character video ID string or None if not found.
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def load_youtube_hybrid(url: str, chunk_seconds: int = 120) -> List[Document]:
    """Load YouTube transcript + metadata and return timestamped chunk Documents.

    This avoids certain 400 errors by manually stitching transcript segments.

    Args:
        url: YouTube video URL.
        chunk_seconds: Approx target duration per chunk.

    Returns:
        List of langchain Document objects with transcript segments + metadata.
    """

    video_id = get_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    try:
        # Updated API: fetch returns a FetchedTranscript with .snippets entries
        transcript = YouTubeTranscriptApi().fetch(video_id=video_id)
    except Exception as e:
        raise Exception(f"Failed to get transcript: {e}")

    metadata = {'video_id': video_id, 'url': url}
    try:
        ydl_opts = {'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            metadata.update({
                'title': info.get('title', f'Video {video_id}'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'upload_date': info.get('upload_date', ''),
                'description': info.get('description', '')[:500]
            })
    except:
        metadata['title'] = f'Video {video_id}'

    chunks = []
    current_chunk = {'text': '', 'start_time': 0, 'end_time': 0}
    # Updated API iteration: iterate over transcript.snippets with attribute access
    for entry in transcript.snippets:
        if entry.start - current_chunk['start_time'] >= chunk_seconds and current_chunk['text']:
            chunks.append(current_chunk.copy())
            current_chunk = {
                'text': entry.text,
                'start_time': entry.start,
                'end_time': entry.start + entry.duration
            }
        else:
            current_chunk['text'] += ' ' + entry.text
            current_chunk['end_time'] = entry.start + entry.duration
    if current_chunk['text']:
        chunks.append(current_chunk)

    documents = []
    for i, chunk in enumerate(chunks):
        start_min = int(chunk['start_time'] // 60)
        start_sec = int(chunk['start_time'] % 60)
        documents.append(Document(
            page_content=chunk['text'].strip(),
            metadata={
                **metadata,
                'chunk_index': i,
                'start_seconds': chunk['start_time'],
                'timestamp': f"{start_min:02d}:{start_sec:02d}",
                'doc_type': 'youtube'
            }
        ))
    return documents


def get_documents(source, doc_type):
    if doc_type == "youtube":
        documents = load_youtube_hybrid(source, chunk_seconds=120)
        first_meta = documents[0].metadata
        title = first_meta.get('title', 'Unknown_Video')
        uploader = first_meta.get('uploader', 'unknown')
        doc_key = build_doc_key('youtube', title, uploader)
    elif doc_type == "pdf":
        loader = PyPDFLoader(source)
        documents = loader.load()
        title = os.path.basename(source).replace('.pdf', '')
        uploader = None
        doc_key = build_doc_key('pdf', title)
    else:
        return f"Error: Unsupported document type '{doc_type}'"
    return (documents, title, uploader, doc_key)


def embed_documents(doc_info, text_splitter, vector_store, metadata, doc_type):
    (documents, title, uploader, doc_key) = doc_info
    chunks = text_splitter.split_documents(documents) if doc_type == "pdf" else documents
    if doc_key in metadata:
        old_num = metadata[doc_key]['num_chunks']
        old_ids = [f"{doc_key}__{i:04d}" for i in range(old_num)]
        try:
            vector_store.delete(ids=old_ids)
        except Exception:
            pass

    chunk_ids = []
    for i, chunk in enumerate(chunks):
        chunk.metadata['doc_key'] = doc_key
        chunk.metadata['title'] = title
        if doc_type == 'youtube':
            chunk.metadata['uploader'] = uploader
        chunk.metadata['doc_type'] = doc_type
        chunk_ids.append(f"{doc_key}__{i:04d}")
    vector_store.add_documents(chunks, ids=chunk_ids)

    return doc_info, chunks


def delete_documents_from_store(metadata, doc_keys, vector_store):
    not_found, deleted_keys, deleted_title_pairs = [], [], []
    for key in doc_keys:
        if key not in metadata:
            not_found.append(key)
            continue
        num = metadata[key]['num_chunks']
        chunk_ids = [f"{key}__{i:04d}" for i in range(num)]
        try:
            vector_store.delete(ids=chunk_ids)
        except Exception:
            pass
        deleted_title_pairs.append((key, metadata[key]['title']))
        deleted_keys.append(key)
        del metadata[key]
    
    return (not_found, deleted_keys, deleted_title_pairs)


def create_deletion_summary(info, current_selection):
    not_found, deleted_keys, deleted_title_pairs = info
    lines = ["Deletion summary:"]
    if deleted_title_pairs:
        lines.append("Deleted:")
        for k, title in deleted_title_pairs:
            lines.append(f" - {title} (key: {k})")
    if not_found:
        lines.append("Not found:")
        for k in not_found:
            lines.append(f" - {k}")
    updated_selection = [k for k in current_selection if k not in deleted_keys]
    summary = "\n".join(lines)
    return summary, updated_selection


def parse_retrieval(results):
    parts = [f"Retrieved {len(results)} chunk(s):"]
    for i, doc in enumerate(results, 1):
        if doc.metadata.get('doc_type') == 'youtube':
            title = doc.metadata.get('title', 'Video')
            ts = doc.metadata.get('timestamp', '??:??')
            source = f"YouTube: {title} @ {ts}"
        else:
            title = doc.metadata.get('title', 'Document')
            page = doc.metadata.get('page', '?')
            source = f"PDF: {title} p{page}"
        snippet = doc.page_content.strip().replace('\n', ' ')
        parts.append(f"\n--- Chunk {i} ({source}) ---\n{snippet}")
    
    return "\n".join(parts)


def create_selection_summary(metadata, doc_keys):
    valid, invalid, titles = [], [], []
    for key in doc_keys:
        if key in metadata:
            valid.append(key)
            titles.append(metadata[key]['title'])
        else:
            invalid.append(key)
    parts = []
    if valid:
        parts.append(f"Selected for chat: {', '.join(titles)}")
    if invalid:
        parts.append(f"Invalid keys: {', '.join(invalid)}")
    summary = "\n".join(parts) if parts else "Deselected all documents from chat."
    return summary, valid


def get_database_info(metadata):
    items = sorted(metadata.items(), key=lambda x: x[1]['title'].lower())
    lines = ["Documents in database:\n"]
    for doc_key, info in items:
        emoji = "📹" if info['type'] == 'youtube' else "📄"
        uploader_part = f" ({info['uploader']})" if info['type'] == 'youtube' and info.get('uploader') else ""
        lines.append(f"{emoji} {info['title']}{uploader_part}")
        lines.append(f"   key: {doc_key} | chunks: {info['num_chunks']} | added: {info['embedded_at'][:10]}")
        lines.append("")
    return "\n".join(lines).strip()
