import re
from typing import Optional, List, Generator

class StreamChunker:
    """
    Accumulates tokens from an LLM stream and yields meaningful chunks (sentences/clauses)
    suitable for TTS processing.
    """
    def __init__(self):
        self.buffer = ""
        # Sentence delimiters: 
        # . ? ! (Standard)
        # | (ASCII Pipe - sometimes used)
        # \u0964 (Devanagari Danda । - Standard Hindi/Indic Full Stop)
        # \n (Newline)
        self.delimiters = re.compile(r'([.?!|\u0964\n]+)')

    def process(self, token: str) -> Generator[str, None, None]:
        """
        Process a new token and yield any complete chunks found.
        
        Args:
            token: New string fragment from LLM
            
        Yields:
            Complete sentence chunks
        """
        self.buffer += token
        
        # Check if buffer contains any delimiter
        # We look for delimiters followed by space or end of string logic roughly
        # Actually simplest is to split by delimiter, keep delimiter with the chunk
        
        while True:
            match = self.delimiters.search(self.buffer)
            if match:
                end_idx = match.end()
                
                # Extract the chunk including the delimiter
                chunk = self.buffer[:end_idx].strip()
                
                if chunk:
                    yield chunk
                
                # Remove processed part from buffer
                self.buffer = self.buffer[end_idx:]
            else:
                break
                
    def flush(self) -> Optional[str]:
        """
        Return any remaining text in buffer as final chunk.
        """
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None
