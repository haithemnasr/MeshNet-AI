"""
Unit tests for core mesh network components.
Run with: pytest tests/ -v
"""
import pytest
from src.network.node import MeshNode
from src.network.network import MeshNetwork
from src.network.message import Message
from src.network.routing import FloodingRouter


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def two_node_network():
    """Minimal network: two nodes within range of each other."""
    net = MeshNetwork()
    n1 = MeshNode("A", x=0,   y=0,   transmission_range=200, initial_battery=100)
    n2 = MeshNode("B", x=100, y=0,   transmission_range=200, initial_battery=100)
    net.add_node(n1)
    net.add_node(n2)
    return net


@pytest.fixture
def three_node_chain():
    """Three nodes in a chain: A -- B -- C."""
    net = MeshNetwork()
    net.add_node(MeshNode("A", x=0,   y=0, transmission_range=150, initial_battery=100))
    net.add_node(MeshNode("B", x=100, y=0, transmission_range=150, initial_battery=100))
    net.add_node(MeshNode("C", x=200, y=0, transmission_range=150, initial_battery=100))
    return net


# ── MeshNode tests ────────────────────────────────────────────────────────────

class TestMeshNode:
    def test_distance_calculation(self):
        a = MeshNode("A", 0, 0, 100)
        b = MeshNode("B", 3, 4, 100)
        assert abs(a.distance_to(b) - 5.0) < 1e-9

    def test_can_reach_within_range(self):
        a = MeshNode("A", 0, 0, 200, initial_battery=100)
        b = MeshNode("B", 100, 0, 200, initial_battery=100)
        assert a.can_reach(b)

    def test_cannot_reach_out_of_range(self):
        a = MeshNode("A", 0, 0, 50, initial_battery=100)
        b = MeshNode("B", 100, 0, 50, initial_battery=100)
        assert not a.can_reach(b)

    def test_inactive_node_unreachable(self):
        a = MeshNode("A", 0, 0, 200, initial_battery=100)
        b = MeshNode("B", 10, 0, 200, initial_battery=100)
        b.is_active = False
        assert not a.can_reach(b)

    def test_consume_energy_kills_node(self):
        node = MeshNode("X", 0, 0, 100, initial_battery=5.0)
        node.consume_energy(10.0)
        assert node.battery == 0
        assert not node.is_active

    def test_add_remove_neighbor(self):
        node = MeshNode("A", 0, 0, 100)
        node.add_neighbor("B")
        assert "B" in node.neighbors
        node.remove_neighbor("B")
        assert "B" not in node.neighbors


# ── MeshNetwork tests ─────────────────────────────────────────────────────────

class TestMeshNetwork:
    def test_nodes_added(self, two_node_network):
        assert len(two_node_network.nodes) == 2

    def test_neighbors_auto_detected(self, two_node_network):
        assert "B" in two_node_network.nodes["A"].neighbors
        assert "A" in two_node_network.nodes["B"].neighbors

    def test_get_active_nodes(self, two_node_network):
        assert len(two_node_network.get_active_nodes()) == 2
        two_node_network.nodes["A"].is_active = False
        assert len(two_node_network.get_active_nodes()) == 1

    def test_send_message_creates_message(self, two_node_network):
        msg = two_node_network.send_message("A", "B", "hello")
        assert msg is not None
        assert msg.source_id == "A"
        assert msg.destination_id == "B"

    def test_chain_neighbors(self, three_node_chain):
        # A should see B, not C; C should see B, not A
        assert "B" in three_node_chain.nodes["A"].neighbors
        assert "C" not in three_node_chain.nodes["A"].neighbors


# ── Message tests ─────────────────────────────────────────────────────────────

class TestMessage:
    def test_initial_path(self):
        msg = Message("1", "A", "B", "hi", timestamp=0.0)
        assert msg.path == ["A"]
        assert msg.hops == 0

    def test_add_hop(self):
        msg = Message("1", "A", "C", "hi", timestamp=0.0)
        msg.add_hop("B")
        msg.add_hop("C")
        assert msg.hops == 2
        assert msg.path == ["A", "B", "C"]


# ── FloodingRouter tests ──────────────────────────────────────────────────────

class TestFloodingRouter:
    def test_message_delivered_two_hops(self, three_node_chain):
        router = FloodingRouter(three_node_chain)
        msg = three_node_chain.send_message("A", "C", "test")

        for _ in range(10):
            router.route_step()

        assert "C" in msg.path

    def test_energy_consumed_on_forward(self, two_node_network):
        router = FloodingRouter(two_node_network)
        two_node_network.send_message("A", "B", "test")
        initial = two_node_network.nodes["A"].battery

        router.route_step()
        assert two_node_network.nodes["A"].battery < initial
