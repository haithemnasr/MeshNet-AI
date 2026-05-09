import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class Message:
    """Represents a message in the mesh network"""
    id: str
    source_id: str
    destination_id: str
    content: str
    timestamp: float
    hops: int = 0
    path: list = None  # Track which nodes the message visited
    
    def __post_init__(self):
        if self.path is None:
            self.path = [self.source_id]
    
    def add_hop(self, node_id: str):
        """Record that this message passed through a node"""
        self.hops += 1
        self.path.append(node_id)
    
    def __repr__(self):
        return f"Message({self.id}: {self.source_id}→{self.destination_id}, hops={self.hops})"