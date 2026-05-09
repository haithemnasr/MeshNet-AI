from typing import Dict, List, Optional
import time
from src.network.node import MeshNode
from src.network.message import Message

class MeshNetwork:
    """Manages the entire mesh network simulation"""
    
    def __init__(self):
        self.nodes: Dict[str, MeshNode] = {}
        self.messages_log: List[Message] = []
        self.simulation_time = 0.0
    
    def add_node(self, node: MeshNode):
        """Add a node to the network"""
        self.nodes[node.id] = node
        self._update_neighbors(node)
    
    def _update_neighbors(self, new_node: MeshNode):
        """Update neighbor lists for all nodes"""
        for node_id, node in self.nodes.items():
            if node_id == new_node.id:
                continue
            
            if new_node.can_reach(node):
                new_node.add_neighbor(node_id)
                node.add_neighbor(new_node.id)
    
    def update_all_neighbors(self):
        """Recalculate all neighbor connections (call after node failures)"""
        # Clear all neighbors
        for node in self.nodes.values():
            node.neighbors.clear()
        
        # Rebuild neighbor lists
        node_list = list(self.nodes.values())
        for i, node1 in enumerate(node_list):
            for node2 in node_list[i+1:]:
                if node1.can_reach(node2):
                    node1.add_neighbor(node2.id)
                    node2.add_neighbor(node1.id)
    
    def send_message(self, source_id: str, dest_id: str, content: str) -> Optional[Message]:
        """Create and send a message from source to destination"""
        if source_id not in self.nodes or dest_id not in self.nodes:
            print(f"Error: Node {source_id} or {dest_id} doesn't exist")
            return None
        
        message = Message(
            id=f"msg_{len(self.messages_log)}",
            source_id=source_id,
            destination_id=dest_id,
            content=content,
            timestamp=self.simulation_time
        )
        
        self.nodes[source_id].send_message(message)
        self.messages_log.append(message)
        return message
    
    def get_active_nodes(self) -> List[MeshNode]:
        """Return list of nodes that are still active"""
        return [node for node in self.nodes.values() if node.is_active]
    
    def get_network_stats(self) -> dict:
        """Get statistics about the network"""
        active_nodes = self.get_active_nodes()
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'dead_nodes': len(self.nodes) - len(active_nodes),
            'total_messages': len(self.messages_log),
            'avg_battery': sum(n.battery for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
        }
    
    def __repr__(self):
        stats = self.get_network_stats()
        return f"MeshNetwork({stats['active_nodes']}/{stats['total_nodes']} nodes active, {stats['total_messages']} messages)"