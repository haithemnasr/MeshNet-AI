import math
from typing import Dict, List, Set, Optional
from src.network.message import Message

class MeshNode:
    """Represents a single node in the mesh network"""
    
    def __init__(self, node_id: str, x: float, y: float, 
                 transmission_range: float = 260.0, 
                 initial_battery: float = 100.0):
        """
        Args:
            node_id: Unique identifier
            x, y: Position coordinates
            transmission_range: How far signals can reach (meters)
            initial_battery: Starting battery percentage
        """
        self.id = node_id
        self.x = x
        self.y = y
        self.transmission_range = transmission_range
        self.battery = initial_battery
        self.is_active = True
        
        # Network state
        self.neighbors: Set[str] = set()  # IDs of reachable nodes
        self.message_queue: List[Message] = []  # Messages to send
        self.received_messages: Set[str] = set()  # Prevent duplicates
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_forwarded = 0
    
    def distance_to(self, other_node: 'MeshNode') -> float:
        """Calculate Euclidean distance to another node"""
        return math.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)
    
    def can_reach(self, other_node: 'MeshNode') -> bool:
        """Check if another node is within transmission range"""
        if not self.is_active or not other_node.is_active:
            return False
        return self.distance_to(other_node) <= self.transmission_range
    
    def add_neighbor(self, neighbor_id: str):
        """Add a node to the neighbor list"""
        self.neighbors.add(neighbor_id)
    
    def remove_neighbor(self, neighbor_id: str):
        """Remove a node from the neighbor list"""
        self.neighbors.discard(neighbor_id)
    
    def send_message(self, message: Message):
        """Add a message to the send queue"""
        self.message_queue.append(message)
        self.messages_sent += 1
    
    def receive_message(self, message: Message) -> bool:
        """
        Receive a message. Returns True if this is a new message.
        """
        if message.id in self.received_messages:
            return False  # Already seen this message
        
        self.received_messages.add(message.id)
        self.messages_received += 1
        message.add_hop(self.id)
        
        return True
    
    def consume_energy(self, amount: float):
        """Reduce battery by specified amount"""
        self.battery = max(0, self.battery - amount)
        if self.battery <= 0:
            self.is_active = False
    
    def __repr__(self):
        status = "ACTIVE" if self.is_active else "DEAD"
        return f"Node({self.id} at ({self.x:.1f},{self.y:.1f}), battery={self.battery:.1f}%, {status})"