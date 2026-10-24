#shared data model for document chunks used throughout the pipeline

from __future__ import annotations
from dataclasses import dataclass,field

@dataclass
class Chunk:
    """
    a single unit of text produced by any splitting strategy
    
    attributes:
        content: the raw text content pf this chunk.
        metadata: key value pairs describing provenance (doc name, file type,page no, chunk index,word count,etc)
        chunk_id: unique,deterministic identifer (hex hash+position)
    """
    content: str
    metadata: dict=field(default_factory=dict)
    chunk_id:str=""
    
    def __post_init__(self)->None:
        if not self.content or not self.content.strip():
            raise ValueError("Chunk content must not be empty")
        
        #ensure word_count is always reflected in metadata
        self.metadata.setdefault('word_count', len(self.content.split()))
        self.metadata.setdefault('char_count', len(self.content))