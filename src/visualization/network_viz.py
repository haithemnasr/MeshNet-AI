import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Optional
from src.network.network import MeshNetwork
from src.network.message import Message
import os

class NetworkVisualizer:
    """Visualize mesh network using matplotlib and NetworkX"""
    
    def __init__(self, network: MeshNetwork, output_dir: str = "data/results"):
        """
        Args:
            network: MeshNetwork instance to visualize
            output_dir: Directory to save images
        """
        self.network = network
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create NetworkX graph
        self.graph = nx.Graph()
        self._build_graph()
    
    def _build_graph(self):
        """Build NetworkX graph from mesh network"""
        # Add nodes with positions
        for node_id, node in self.network.nodes.items():
            self.graph.add_node(
                node_id, 
                pos=(node.x, node.y),
                battery=node.battery,
                active=node.is_active
            )
        
        # Add edges (connections between neighbors)
        for node_id, node in self.network.nodes.items():
            for neighbor_id in node.neighbors:
                # Only add edge once (undirected graph)
                if node_id < neighbor_id:
                    self.graph.add_edge(node_id, neighbor_id)
    
    def plot_network(self, title: str = "Mesh Network Topology", 
                     filename: str = "network_topology.png",
                     figsize: tuple = (12, 10)):
        """
        Plot the basic network topology
        
        Args:
            title: Plot title
            filename: Output filename
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Get node positions
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Get battery levels for coloring
        batteries = nx.get_node_attributes(self.graph, 'battery')
        node_colors = [batteries[node] for node in self.graph.nodes()]
        
        # Draw the network
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=800,
            cmap=plt.cm.RdYlGn,  # Red (low) to Green (high)
            vmin=0, vmax=100,
            edgecolors='black',
            linewidths=2
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            width=2,
            alpha=0.5
        )
        
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=10,
            font_weight='bold'
        )
        
        # Add colorbar for battery levels
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.RdYlGn,
            norm=plt.Normalize(vmin=0, vmax=100)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Battery Level (%)')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        
        plt.show()
    
    def plot_message_path(self, message: Message, 
                          title: str = None,
                          filename: str = "message_path.png",
                          figsize: tuple = (12, 10)):
        """
        Highlight the path a message took through the network
        
        Args:
            message: Message object with path information
            title: Plot title (auto-generated if None)
            filename: Output filename
            figsize: Figure size
        """
        if title is None:
            title = f"Message Path: {message.source_id} → {message.destination_id}"
        
        plt.figure(figsize=figsize)
        
        # Get node positions
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Color nodes: path nodes in blue, others in light gray
        node_colors = []
        for node in self.graph.nodes():
            if node in message.path:
                node_colors.append('royalblue')
            else:
                node_colors.append('lightgray')
        
        # Draw all nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=800,
            edgecolors='black',
            linewidths=2
        )
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='lightgray',
            width=2,
            alpha=0.3
        )
        
        # Highlight the path edges in red
        path_edges = [(message.path[i], message.path[i+1]) 
                      for i in range(len(message.path)-1)]
        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=path_edges,
            edge_color='red',
            width=4,
            alpha=0.8,
            arrows=True,
            arrowsize=20,
            arrowstyle='->'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=10,
            font_weight='bold'
        )
        
        # Add path info text
        path_text = " → ".join(message.path)
        plt.text(
            0.5, 0.02, 
            f"Path ({message.hops} hops): {path_text}",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            ha='center'
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        
        plt.show()
    
    def plot_battery_levels(self, title: str = "Node Battery Levels",
                           filename: str = "battery_levels.png",
                           figsize: tuple = (12, 10)):
        """
        Plot network with nodes colored by battery level
        
        Args:
            title: Plot title
            filename: Output filename
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Get node positions and batteries
        pos = nx.get_node_attributes(self.graph, 'pos')
        batteries = nx.get_node_attributes(self.graph, 'battery')
        
        node_colors = [batteries[node] for node in self.graph.nodes()]
        
        # Draw nodes with battery coloring
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=1000,
            cmap=plt.cm.RdYlGn,
            vmin=0, vmax=100,
            edgecolors='black',
            linewidths=2
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            width=2,
            alpha=0.5
        )
        
        # Draw labels with battery percentage
        labels = {node: f"{node}\n{batteries[node]:.1f}%" 
                  for node in self.graph.nodes()}
        nx.draw_networkx_labels(
            self.graph, pos,
            labels=labels,
            font_size=8,
            font_weight='bold'
        )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.RdYlGn,
            norm=plt.Normalize(vmin=0, vmax=100)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Battery Level (%)')
        
        # Add statistics text
        avg_battery = sum(batteries.values()) / len(batteries)
        stats_text = f"Average Battery: {avg_battery:.1f}%\n"
        stats_text += f"Active Nodes: {len(self.network.get_active_nodes())}/{len(self.network.nodes)}"
        
        plt.text(
            0.02, 0.98,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        
        plt.show()
    
    def plot_statistics(self, filename: str = "network_stats.png",
                       figsize: tuple = (14, 8)):
        """
        Create a dashboard with multiple network statistics
        
        Args:
            filename: Output filename
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Network Statistics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Battery distribution
        batteries = [node.battery for node in self.network.nodes.values()]
        axes[0, 0].hist(batteries, bins=10, color='green', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Battery Level (%)')
        axes[0, 0].set_ylabel('Number of Nodes')
        axes[0, 0].set_title('Battery Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Messages per node
        node_ids = [node.id for node in self.network.nodes.values()]
        messages_forwarded = [node.messages_forwarded for node in self.network.nodes.values()]
        axes[0, 1].bar(node_ids, messages_forwarded, color='royalblue', edgecolor='black')
        axes[0, 1].set_xlabel('Node ID')
        axes[0, 1].set_ylabel('Messages Forwarded')
        axes[0, 1].set_title('Message Traffic per Node')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Neighbor count distribution
        neighbor_counts = [len(node.neighbors) for node in self.network.nodes.values()]
        axes[1, 0].hist(neighbor_counts, bins=range(max(neighbor_counts)+2), 
                       color='orange', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Number of Neighbors')
        axes[1, 0].set_ylabel('Number of Nodes')
        axes[1, 0].set_title('Network Connectivity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Network summary
        axes[1, 1].axis('off')
        stats = self.network.get_network_stats()
        summary_text = f"""
        Network Summary
        ━━━━━━━━━━━━━━━━━━━━━━
        Total Nodes: {stats['total_nodes']}
        Active Nodes: {stats['active_nodes']}
        Dead Nodes: {stats['dead_nodes']}
        
        Total Messages: {stats['total_messages']}
        Avg Battery: {stats['avg_battery']:.1f}%
        
        Connectivity:
        Min Neighbors: {min(neighbor_counts)}
        Max Neighbors: {max(neighbor_counts)}
        Avg Neighbors: {sum(neighbor_counts)/len(neighbor_counts):.1f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                       family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        
        plt.show()