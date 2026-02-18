"""
Snapshot Loader - Phase C
Reads transcript from a file.
"""

import os

def load_snapshot(path: str) -> str:
    """
    Load transcript text from a file.

    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    return content
