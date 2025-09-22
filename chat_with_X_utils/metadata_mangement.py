import json
import os
import tempfile
from typing import Dict, Callable, Tuple, Any
import threading
import fcntl

# In-process lock (threads / async tasks in same interpreter)
_INPROCESS_METADATA_LOCK = threading.Lock()

def _acquire_file_lock(fp):
    """Acquire an OS-level advisory lock on an open file descriptor (POSIX)."""
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
    except Exception:
        pass

def _release_file_lock(fp):
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass


def load_metadata(metadata_file: str) -> Dict:
    """Load document metadata from file.

    Note: This is a non-locking read. For write-modify-write cycles use
    `atomic_update_metadata` to avoid lost updates.
    """
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_metadata(metadata: Dict, save_file: str):
    """Persist document metadata to file atomically (best effort)."""
    directory = os.path.dirname(save_file) or "."
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=directory, delete=False) as tmp:
        json.dump(metadata, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno()) if hasattr(os, 'fsync') else None
        temp_name = tmp.name
    os.replace(temp_name, save_file)


def atomic_update_metadata(metadata_file: str, update_fn: Callable[[Dict], Tuple[Dict, Any]]) -> Any:
    """Safely update metadata with combined process + thread locking.

    Args:
        metadata_file: Path to JSON metadata file.
        update_fn: Callable taking current dict and returning (updated_dict, result)
            where `result` is any value you want returned to caller.

    Returns:
        The `result` from update_fn.
    """
    directory = os.path.dirname(metadata_file) or "."
    os.makedirs(directory, exist_ok=True)
    # Open file (create if missing) for locking
    with _INPROCESS_METADATA_LOCK:
        # Use separate lock file to avoid interfering with atomic rename pattern
        lock_path = metadata_file + ".lock"
        with open(lock_path, 'a+') as lock_fp:
            _acquire_file_lock(lock_fp)
            try:
                current = load_metadata(metadata_file)
                updated, result = update_fn(current)
                save_metadata(updated, metadata_file)
            finally:
                _release_file_lock(lock_fp)
    return result
