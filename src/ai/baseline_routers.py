"""
Baseline routing algorithms for mesh networks
Used for comparing against RL agents
"""

import heapq
from typing import Dict, List, Set, Optional, Tuple
from src.network.network import MeshNetwork
from src.network.message import Message
from src.network.node import MeshNode

class BaseRouter:
    """Base class for routing algorithms"""
    
    def __init__(self, network: MeshNetwork, energy_per_transmission: float = 0.1):
        self.network = network
        self.energy_per_transmission = energy_per_transmission
        self.messages_delivered = 0
        self.messages_dropped = 0
        self.total_energy_used = 0.0
        self.total_hops = 0
    
    def route_message(self, message: Message) -> bool:
        """
        Route a single message. Returns True if delivered.
        Must be implemented by subclass.
        """
        raise NotImplementedError
    
    def get_statistics(self) -> Dict:
        """Get routing statistics"""
        total = self.messages_delivered + self.messages_dropped
        return {
            'delivered': self.messages_delivered,
            'dropped': self.messages_dropped,
            'total': total,
            'delivery_rate': self.messages_delivered / total if total > 0 else 0,
            'total_energy': self.total_energy_used,
            'avg_energy': self.total_energy_used / total if total > 0 else 0,
            'avg_hops': self.total_hops / self.messages_delivered if self.messages_delivered > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.messages_delivered = 0
        self.messages_dropped = 0
        self.total_energy_used = 0.0
        self.total_hops = 0


class ShortestPathRouter(BaseRouter):
    """
    Dijkstra's shortest path algorithm
    Finds optimal path based on hop count
    """
    
    def __init__(self, network: MeshNetwork, energy_per_transmission: float = 0.1):
        super().__init__(network, energy_per_transmission)
        self.name = "Shortest Path (Dijkstra)"
    
    def _find_shortest_path(self, source_id: str, dest_id: str) -> Optional[List[str]]:
        """
        Find shortest path using Dijkstra's algorithm
        
        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        # Priority queue: (distance, node_id, path)
        pq = [(0, source_id, [source_id])]
        visited = set()
        
        while pq:
            dist, current_id, path = heapq.heappop(pq)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Found destination
            if current_id == dest_id:
                return path
            
            # Explore neighbors
            current_node = self.network.nodes[current_id]
            for neighbor_id in current_node.neighbors:
                neighbor_node = self.network.nodes[neighbor_id]
                
                # Only consider active nodes
                if not neighbor_node.is_active or neighbor_id in visited:
                    continue
                
                new_path = path + [neighbor_id]
                new_dist = dist + 1  # Hop count
                heapq.heappush(pq, (new_dist, neighbor_id, new_path))
        
        return None  # No path found
    
    def route_message(self, message: Message) -> bool:
        """Route message along shortest path"""
        path = self._find_shortest_path(message.source_id, message.destination_id)
        
        if path is None:
            # No path exists
            self.messages_dropped += 1
            return False
        
        # Simulate transmission along path
        for i in range(len(path) - 1):
            current_node = self.network.nodes[path[i]]
            next_node = self.network.nodes[path[i + 1]]
            
            # Calculate energy cost based on distance
            distance = current_node.distance_to(next_node)
            energy_cost = self.energy_per_transmission + (distance / 100.0) * 0.05
            
            current_node.consume_energy(energy_cost)
            self.total_energy_used += energy_cost
            message.add_hop(path[i + 1])
            
            # Check if node died during transmission
            if not current_node.is_active:
                self.messages_dropped += 1
                return False
        
        self.messages_delivered += 1
        self.total_hops += len(path) - 1
        return True


class GreedyRouter(BaseRouter):
    """
    Greedy routing: Always choose neighbor closest to destination
    """
    
    def __init__(self, network: MeshNetwork, energy_per_transmission: float = 0.1):
        super().__init__(network, energy_per_transmission)
        self.name = "Greedy (Closest Neighbor)"
        self.max_hops = 50
    
    def route_message(self, message: Message) -> bool:
        """Route message greedily toward destination"""
        current_id = message.source_id
        dest_node = self.network.nodes[message.destination_id]
        
        visited = {current_id}
        hops = 0
        
        while current_id != message.destination_id and hops < self.max_hops:
            current_node = self.network.nodes[current_id]
            
            # Find neighbor closest to destination
            best_neighbor = None
            best_distance = float('inf')
            
            for neighbor_id in current_node.neighbors:
                neighbor_node = self.network.nodes[neighbor_id]
                
                # Skip inactive or visited nodes
                if not neighbor_node.is_active or neighbor_id in visited:
                    continue
                
                distance = neighbor_node.distance_to(dest_node)
                if distance < best_distance:
                    best_distance = distance
                    best_neighbor = neighbor_id
            
            # No valid neighbor found
            if best_neighbor is None:
                self.messages_dropped += 1
                return False
            
            # Move to next node
            next_node = self.network.nodes[best_neighbor]
            distance = current_node.distance_to(next_node)
            energy_cost = self.energy_per_transmission + (distance / 100.0) * 0.05
            
            current_node.consume_energy(energy_cost)
            self.total_energy_used += energy_cost
            message.add_hop(best_neighbor)
            
            visited.add(best_neighbor)
            current_id = best_neighbor
            hops += 1
        
        if current_id == message.destination_id:
            self.messages_delivered += 1
            self.total_hops += hops
            return True
        else:
            self.messages_dropped += 1
            return False


class AODVRouter(BaseRouter):
    """
    AODV (Ad-hoc On-Demand Distance Vector) routing
    Discovers routes on-demand and maintains routing tables
    """
    
    def __init__(self, network: MeshNetwork, energy_per_transmission: float = 0.1):
        super().__init__(network, energy_per_transmission)
        self.name = "AODV (On-Demand)"
        self.routing_tables: Dict[str, Dict[str, Tuple[str, int]]] = {}  # node -> {dest: (next_hop, distance)}
        self.max_hops = 50
    
    def _route_discovery(self, source_id: str, dest_id: str) -> Optional[List[str]]:
        """
        Simulate AODV route discovery (RREQ/RREP)
        Simplified version using BFS
        """
        # BFS to find path
        queue = [(source_id, [source_id])]
        visited = {source_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == dest_id:
                # Update routing tables along path
                for i in range(len(path) - 1):
                    node_id = path[i]
                    if node_id not in self.routing_tables:
                        self.routing_tables[node_id] = {}
                    
                    # Add route to destination
                    next_hop = path[i + 1]
                    distance = len(path) - i - 1
                    self.routing_tables[node_id][dest_id] = (next_hop, distance)
                
                return path
            
            current_node = self.network.nodes[current_id]
            for neighbor_id in current_node.neighbors:
                neighbor_node = self.network.nodes[neighbor_id]
                
                if neighbor_node.is_active and neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return None
    
    def route_message(self, message: Message) -> bool:
        """Route message using AODV protocol"""
        current_id = message.source_id
        dest_id = message.destination_id
        
        # Check if route exists in table
        if (current_id in self.routing_tables and 
            dest_id in self.routing_tables[current_id]):
            # Use cached route
            next_hop, _ = self.routing_tables[current_id][dest_id]
        else:
            # Discover new route
            path = self._route_discovery(current_id, dest_id)
            if path is None:
                self.messages_dropped += 1
                return False
            next_hop = path[1] if len(path) > 1 else dest_id
        
        # Forward message hop by hop
        hops = 0
        while current_id != dest_id and hops < self.max_hops:
            current_node = self.network.nodes[current_id]
            
            # Get next hop from routing table
            if current_id not in self.routing_tables or dest_id not in self.routing_tables[current_id]:
                # Route broken, rediscover
                path = self._route_discovery(current_id, dest_id)
                if path is None:
                    self.messages_dropped += 1
                    return False
                next_hop = path[1]
            else:
                next_hop, _ = self.routing_tables[current_id][dest_id]
            
            # Check if next hop is still active
            next_node = self.network.nodes[next_hop]
            if not next_node.is_active:
                # Route broken, invalidate and rediscover
                del self.routing_tables[current_id][dest_id]
                path = self._route_discovery(current_id, dest_id)
                if path is None:
                    self.messages_dropped += 1
                    return False
                next_hop = path[1]
                next_node = self.network.nodes[next_hop]
            
            # Transmit to next hop
            distance = current_node.distance_to(next_node)
            energy_cost = self.energy_per_transmission + (distance / 100.0) * 0.05
            
            current_node.consume_energy(energy_cost)
            self.total_energy_used += energy_cost
            message.add_hop(next_hop)
            
            current_id = next_hop
            hops += 1
        
        if current_id == dest_id:
            self.messages_delivered += 1
            self.total_hops += hops
            return True
        else:
            self.messages_dropped += 1
            return False