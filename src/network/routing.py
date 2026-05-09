from typing import Dict, List
from src.network.network import MeshNetwork  # Changed path
from src.network.message import Message

class FloodingRouter:
    """Simple flooding algorithm: broadcast to all neighbors"""
    
    def __init__(self, network: MeshNetwork):
        self.network = network
        self.energy_per_transmission = 0.1  # Battery % consumed per message
    
    def route_step(self):
        """
        Execute one routing step: 
        Each node forwards its queued messages to all neighbors
        """
        messages_to_forward = []
        
        # Collect all messages that need forwarding
        for node in self.network.get_active_nodes():
            while node.message_queue:
                msg = node.message_queue.pop(0)
                
                # Check if this node is the destination
                if node.id == msg.destination_id:
                    print(f"✓ Message {msg.id} DELIVERED to {node.id} in {msg.hops} hops!")
                    continue
                
                # Forward to all neighbors
                for neighbor_id in node.neighbors:
                    messages_to_forward.append((neighbor_id, msg))
                    node.consume_energy(self.energy_per_transmission)
                    node.messages_forwarded += 1
        
        # Deliver forwarded messages
        for neighbor_id, msg in messages_to_forward:
            neighbor = self.network.nodes[neighbor_id]
            if neighbor.receive_message(msg):
                # Only queue if this is the first time receiving it
                neighbor.message_queue.append(msg)