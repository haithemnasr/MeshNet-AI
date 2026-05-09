"""
Unit tests for baseline routers and RL environment.
"""
import pytest
import numpy as np
from src.network.node import MeshNode
from src.network.network import MeshNetwork
from src.network.message import Message
from src.ai.baseline_routers import ShortestPathRouter, GreedyRouter, AODVRouter
from src.ai.stable_rl_env import StableMeshRoutingEnv


# ── Fixture: small fully-connected network ────────────────────────────────────

@pytest.fixture
def small_network():
    """5 nodes arranged in a star topology."""
    net = MeshNetwork()
    center = MeshNode("center", 500, 500, transmission_range=300, initial_battery=100)
    net.add_node(center)
    positions = [(200, 500), (800, 500), (500, 200), (500, 800)]
    for i, (x, y) in enumerate(positions):
        net.add_node(MeshNode(f"n{i}", x, y, transmission_range=300, initial_battery=100))
    return net


# ── ShortestPathRouter ────────────────────────────────────────────────────────

class TestShortestPathRouter:
    def test_delivers_message(self, small_network):
        router = ShortestPathRouter(small_network)
        msg = Message("m1", "n0", "n1", "test", timestamp=0.0)
        result = router.route_message(msg)
        assert result is True

    def test_statistics_delivery_rate(self, small_network):
        router = ShortestPathRouter(small_network)
        for i in range(5):
            msg = Message(f"m{i}", "n0", "n2", "test", timestamp=0.0)
            router.route_message(msg)
        stats = router.get_statistics()
        assert stats["delivery_rate"] > 0

    def test_drops_when_isolated(self):
        net = MeshNetwork()
        net.add_node(MeshNode("solo", 0, 0, 50, initial_battery=100))
        net.add_node(MeshNode("far",  999, 999, 50, initial_battery=100))
        router = ShortestPathRouter(net)
        msg = Message("m1", "solo", "far", "test", timestamp=0.0)
        result = router.route_message(msg)
        assert result is False


# ── GreedyRouter ──────────────────────────────────────────────────────────────

class TestGreedyRouter:
    def test_delivers_message(self, small_network):
        router = GreedyRouter(small_network)
        msg = Message("m1", "n0", "n1", "test", timestamp=0.0)
        result = router.route_message(msg)
        assert result is True

    def test_statistics_have_hops(self, small_network):
        router = GreedyRouter(small_network)
        msg = Message("m1", "n0", "n3", "test", timestamp=0.0)
        router.route_message(msg)
        stats = router.get_statistics()
        assert stats["avg_hops"] >= 0


# ── AODVRouter ────────────────────────────────────────────────────────────────

class TestAODVRouter:
    def test_delivers_message(self, small_network):
        router = AODVRouter(small_network)
        msg = Message("m1", "n0", "n2", "test", timestamp=0.0)
        result = router.route_message(msg)
        assert result is True

    def test_caches_route(self, small_network):
        router = AODVRouter(small_network)
        msg1 = Message("m1", "n0", "n2", "test", timestamp=0.0)
        router.route_message(msg1)
        # Route should now be cached
        assert "n0" in router.routing_tables


# ── StableMeshRoutingEnv ──────────────────────────────────────────────────────

class TestStableMeshRoutingEnv:
    def test_observation_shape(self, small_network):
        env = StableMeshRoutingEnv(small_network, max_hops=10)
        obs, _ = env.reset()
        assert obs.shape == (StableMeshRoutingEnv.STATE_SIZE,)

    def test_observation_bounds(self, small_network):
        env = StableMeshRoutingEnv(small_network, max_hops=10)
        obs, _ = env.reset()
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0 + 1e-6)

    def test_step_returns_correct_structure(self, small_network):
        env = StableMeshRoutingEnv(small_network, max_hops=10)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (StableMeshRoutingEnv.STATE_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "result" in info

    def test_episode_terminates(self, small_network):
        env = StableMeshRoutingEnv(small_network, max_hops=5)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert done
