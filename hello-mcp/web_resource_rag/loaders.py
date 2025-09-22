from typing import List, Iterator
import re
from urllib.parse import urlparse, parse_qs

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript
import yt_dlp


class YouTubeTranscriptLoader(BaseLoader):
    """Load YouTube transcript + metadata and return timestamped chunk Documents.
    
    This loader avoids certain 400 errors by manually stitching transcript segments
    and provides rich metadata about the video.
    """
    
    def __init__(self, url: str, chunk_seconds: int = 120):
        """Initialize the YouTube Hybrid Loader.
        
        Args:
            url: YouTube video URL.
            chunk_seconds: Approximate target duration per chunk in seconds.
        """
        self.url = url
        self.chunk_seconds = chunk_seconds
        self.video_id = self._get_video_id(url)
        
        if not self.video_id:
            raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def _get_video_id(self, url: str) -> str | None:
        """Extract video ID from YouTube URL."""
        # Handle various YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Try parsing as query parameter
        try:
            parsed = urlparse(url)
            if 'youtube.com' in parsed.netloc:
                return parse_qs(parsed.query).get('v', [None])[0]
            elif 'youtu.be' in parsed.netloc:
                return parsed.path.lstrip('/')
        except:
            pass
        
        return None
    
    def _get_video_metadata(self) -> dict:
        """Extract video metadata using yt-dlp."""
        metadata = {
            'video_id': self.video_id, 
            'url': self.url,
            'title': f'Video {self.video_id}'  # Default fallback
        }
        
        try:
            ydl_opts = {
                'quiet': True, 
                'no_warnings': True,
                'extract_flat': False
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                metadata.update({
                    'title': info.get('title', f'Video {self.video_id}'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', '')[:500],  # Truncate description
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'channel_id': info.get('channel_id', ''),
                    'channel_url': info.get('channel_url', ''),
                })
        except Exception as e:
            # Silently fail and use defaults - metadata extraction is not critical
            pass
        
        return metadata
    
    def _get_transcript(self) -> FetchedTranscript:
        """Fetch transcript using YouTubeTranscriptApi."""
        try:
            assert self.video_id
            transcript = YouTubeTranscriptApi().fetch(video_id=self.video_id)
            return transcript
        except Exception as e:
            raise Exception(f"Failed to get transcript for video {self.video_id}: {e}")
    
    def _create_chunks(self, transcript: FetchedTranscript, metadata: dict) -> List[Document]:
        """Create timestamped chunks from transcript segments."""
        chunks = []
        current_chunk = {'text': '', 'start_time': 0, 'end_time': 0}

        for entry in transcript.snippets:
            entry_start = entry.start
            entry_duration = entry.duration
            entry_text = entry.text
            
            # Check if we should start a new chunk
            if (entry_start - current_chunk['start_time'] >= self.chunk_seconds and 
                current_chunk['text']):
                chunks.append(current_chunk.copy())
                current_chunk = {
                    'text': entry_text,
                    'start_time': entry_start,
                    'end_time': entry_start + entry_duration
                }
            else:
                # Add to current chunk
                if current_chunk['text']:
                    current_chunk['text'] += ' ' + entry_text
                else:
                    current_chunk['text'] = entry_text
                    current_chunk['start_time'] = entry_start
                
                current_chunk['end_time'] = entry_start + entry_duration
        
        # Add the last chunk if it has content
        if current_chunk['text']:
            chunks.append(current_chunk)
        
        # Convert chunks to Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            start_min = int(chunk['start_time'] // 60)
            start_sec = int(chunk['start_time'] % 60)
            end_min = int(chunk['end_time'] // 60)
            end_sec = int(chunk['end_time'] % 60)
            
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'start_seconds': chunk['start_time'],
                'end_seconds': chunk['end_time'],
                'timestamp': f"{start_min:02d}:{start_sec:02d}",
                'timestamp_range': f"{start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}",
                'doc_type': 'youtube',
                'chunk_duration': chunk['end_time'] - chunk['start_time']
            }
            
            documents.append(Document(
                page_content=chunk['text'].strip(),
                metadata=chunk_metadata
            ))
        
        return documents
    
    def load(self) -> List[Document]:
        """Load YouTube transcript and return timestamped chunk Documents.
        
        Returns:
            List of langchain Document objects with transcript segments + metadata.
        """
        # Get video metadata
        metadata = self._get_video_metadata()
        # Get transcript
        transcript = self._get_transcript()
        # Create and return chunked documents
        return self._create_chunks(transcript, metadata)
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load implementation that yields documents one at a time."""
        documents = self.load()
        for doc in documents:
            yield doc


