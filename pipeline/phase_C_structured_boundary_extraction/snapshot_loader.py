"""
Snapshot Loader - Phase C
Reads transcript from a file.
"""

import os

def load_snapshot(path: str) -> str:
    """
    Load transcript text from a file.

    Args:
        path (str): Path to the transcript file.

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    return content
