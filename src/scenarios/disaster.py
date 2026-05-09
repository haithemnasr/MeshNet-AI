"""
Disaster scenario simulator for Tunisian mesh network
Simulates flooding, earthquakes, and infrastructure failures
"""

import random
from typing import List, Set, Dict
from src.network.network import MeshNetwork
from src.network.node import MeshNode
from src.scenarios.tunisia_map import TunisianCity, TERRAIN_TYPES

class DisasterScenario:
    """Base class for disaster scenarios"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.affected_nodes: Set[str] = set()
        self.time_step = 0
    
    def start(self, network: MeshNetwork):
        """Initialize the disaster"""
        raise NotImplementedError
    
    def update(self, network: MeshNetwork):
        """Update disaster effects over time"""
        raise NotImplementedError
    
    def is_active(self) -> bool:
        """Check if disaster is still ongoing"""
        raise NotImplementedError


class FloodingScenario(DisasterScenario):
    """
    Simulates coastal flooding (Nabeul, Bizerte style)
    - Nodes fail progressively based on elevation
    - Spreads from low-lying coastal areas
    """
    
    def __init__(self, epicenter_x: float, epicenter_y: float, 
                 max_radius: float = 200, duration: int = 10):
        """
        Args:
            epicenter_x, epicenter_y: Flood starting point
            max_radius: Maximum flood spread distance
            duration: How many time steps the flood expands
        """
        super().__init__(
            name="Coastal Flooding",
            description=f"Flooding at ({epicenter_x:.0f}, {epicenter_y:.0f})"
        )
        self.epicenter_x = epicenter_x
        self.epicenter_y = epicenter_y
        self.max_radius = max_radius
        self.duration = duration
        self.current_radius = 0
    
    def start(self, network: MeshNetwork):
        """Begin flooding scenario"""
        self.time_step = 0
        self.current_radius = 0
        self.affected_nodes.clear()
        print(f"\n🌊 DISASTER: {self.name} starting at ({self.epicenter_x:.0f}, {self.epicenter_y:.0f})")
    
    def update(self, network: MeshNetwork):
        """Flood expands outward each time step"""
        if not self.is_active():
            return
        
        # Expand flood radius
        self.current_radius = (self.time_step / self.duration) * self.max_radius
        
        newly_affected = []
        for node_id, node in network.nodes.items():
            if not node.is_active or node_id in self.affected_nodes:
                continue
            
            # Calculate distance from epicenter
            distance = ((node.x - self.epicenter_x)**2 + 
                       (node.y - self.epicenter_y)**2)**0.5
            
            # Node fails if within flood radius
            if distance <= self.current_radius:
                # Lower elevation = higher failure chance
                # Simulate that low-lying areas flood first
                failure_chance = 0.7 if distance < self.current_radius * 0.5 else 0.4
                
                if random.random() < failure_chance:
                    node.is_active = False
                    node.battery = 0
                    self.affected_nodes.add(node_id)
                    newly_affected.append(node_id)
        
        if newly_affected:
            print(f"  Step {self.time_step}: Flood radius={self.current_radius:.0f}m, "
                  f"{len(newly_affected)} nodes flooded: {newly_affected}")
        
        self.time_step += 1
        network.update_all_neighbors()
    
    def is_active(self) -> bool:
        return self.time_step < self.duration


class EarthquakeScenario(DisasterScenario):
    """
    Simulates earthquake (Kasserine, mountainous regions)
    - Sudden random node failures
    - Aftershocks cause additional failures
    """
    
    def __init__(self, epicenter_x: float, epicenter_y: float,
                 magnitude: float = 6.0, aftershocks: int = 3):
        """
        Args:
            epicenter_x, epicenter_y: Earthquake epicenter
            magnitude: Earthquake strength (affects failure radius)
            aftershocks: Number of aftershock events
        """
        super().__init__(
            name=f"Earthquake (Magnitude {magnitude})",
            description=f"Seismic event at ({epicenter_x:.0f}, {epicenter_y:.0f})"
        )
        self.epicenter_x = epicenter_x
        self.epicenter_y = epicenter_y
        self.magnitude = magnitude
        self.aftershocks = aftershocks
        self.aftershock_schedule = []
    
    def start(self, network: MeshNetwork):
        """Main earthquake hit"""
        self.time_step = 0
        self.affected_nodes.clear()
        
        print(f"\n🏚️ DISASTER: {self.name} strikes!")
        
        # Calculate damage radius based on magnitude
        damage_radius = self.magnitude * 50
        
        # Main quake damage
        for node_id, node in network.nodes.items():
            distance = ((node.x - self.epicenter_x)**2 + 
                       (node.y - self.epicenter_y)**2)**0.5
            
            # Closer = more damage
            if distance < damage_radius:
                failure_prob = 1.0 - (distance / damage_radius)
                if random.random() < failure_prob:
                    node.is_active = False
                    node.battery = 0
                    self.affected_nodes.add(node_id)
        
        print(f"  Main quake: {len(self.affected_nodes)} nodes destroyed")
        
        # Schedule aftershocks
        self.aftershock_schedule = [random.randint(2, 6) for _ in range(self.aftershocks)]
        self.aftershock_schedule.sort()
        
        network.update_all_neighbors()
    
    def update(self, network: MeshNetwork):
        """Aftershocks at scheduled times"""
        self.time_step += 1
        
        if self.aftershock_schedule and self.time_step >= self.aftershock_schedule[0]:
            self.aftershock_schedule.pop(0)
            
            # Aftershock: random node failures
            active_nodes = [nid for nid, n in network.nodes.items() if n.is_active]
            if active_nodes:
                num_failures = random.randint(1, max(1, len(active_nodes) // 5))
                failed = random.sample(active_nodes, min(num_failures, len(active_nodes)))
                
                for node_id in failed:
                    network.nodes[node_id].is_active = False
                    network.nodes[node_id].battery = 0
                    self.affected_nodes.add(node_id)
                
                print(f"  Aftershock at step {self.time_step}: {len(failed)} nodes damaged")
                network.update_all_neighbors()
    
    def is_active(self) -> bool:
        return len(self.aftershock_schedule) > 0


class InfrastructureFailureScenario(DisasterScenario):
    """
    Simulates infrastructure collapse (Tunis style)
    - Central hub nodes fail suddenly
    - Causes cascade effect on connected nodes
    """
    
    def __init__(self, hub_nodes: List[str], cascade_probability: float = 0.3):
        """
        Args:
            hub_nodes: Node IDs of critical infrastructure
            cascade_probability: Chance that failure spreads to neighbors
        """
        super().__init__(
            name="Infrastructure Collapse",
            description=f"Critical hubs {hub_nodes} failing"
        )
        self.hub_nodes = hub_nodes
        self.cascade_probability = cascade_probability
        self.cascade_active = True
    
    def start(self, network: MeshNetwork):
        """Central infrastructure fails"""
        self.time_step = 0
        self.affected_nodes.clear()
        
        print(f"\n⚠️ DISASTER: {self.name}")
        
        # Fail hub nodes immediately
        for node_id in self.hub_nodes:
            if node_id in network.nodes:
                network.nodes[node_id].is_active = False
                network.nodes[node_id].battery = 0
                self.affected_nodes.add(node_id)
        
        print(f"  Critical hubs failed: {self.hub_nodes}")
        network.update_all_neighbors()
    
    def update(self, network: MeshNetwork):
        """Cascade failures spread to neighbors"""
        if not self.cascade_active:
            return
        
        self.time_step += 1
        
        # Find neighbors of failed nodes
        cascade_candidates = set()
        for failed_node in self.affected_nodes:
            if failed_node in network.nodes:
                # Neighbors of failed nodes are at risk
                neighbors = network.nodes[failed_node].neighbors
                for neighbor_id in neighbors:
                    if network.nodes[neighbor_id].is_active:
                        cascade_candidates.add(neighbor_id)
        
        # Some neighbors fail due to overload
        newly_failed = []
        for node_id in cascade_candidates:
            if random.random() < self.cascade_probability:
                network.nodes[node_id].is_active = False
                network.nodes[node_id].battery = 0
                self.affected_nodes.add(node_id)
                newly_failed.append(node_id)
        
        if newly_failed:
            print(f"  Step {self.time_step}: Cascade failures: {newly_failed}")
            network.update_all_neighbors()
        else:
            self.cascade_active = False
            print(f"  Step {self.time_step}: Cascade stopped")
    
    def is_active(self) -> bool:
        return self.cascade_active


# Predefined Tunisia-specific disaster scenarios
def create_nabeul_flood(network: MeshNetwork) -> FloodingScenario:
    """Flooding in Nabeul coastal area"""
    return FloodingScenario(
        epicenter_x=550,  # Nabeul coordinates
        epicenter_y=180,
        max_radius=150,
        duration=8
    )

def create_tunis_infrastructure_collapse(network: MeshNetwork) -> InfrastructureFailureScenario:
    """Major infrastructure failure in Tunis"""
    # Find central/hub nodes (most connected)
    hub_nodes = sorted(
        network.nodes.keys(),
        key=lambda nid: len(network.nodes[nid].neighbors),
        reverse=True
    )[:3]  # Top 3 most connected nodes
    
    return InfrastructureFailureScenario(
        hub_nodes=hub_nodes,
        cascade_probability=0.25
    )

def create_kasserine_earthquake(network: MeshNetwork) -> EarthquakeScenario:
    """Earthquake in mountainous Kasserine region"""
    return EarthquakeScenario(
        epicenter_x=400,  # Kasserine coordinates
        epicenter_y=550,
        magnitude=5.5,
        aftershocks=4
    )