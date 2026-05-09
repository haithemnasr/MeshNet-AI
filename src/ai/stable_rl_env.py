import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from src.network.network import MeshNetwork
from src.network.message import Message


class StableMeshRoutingEnv(gym.Env):
    """
    Enhanced RL environment with per-neighbor state representation.

    Changes vs previous version:
    1. State expanded: 8 global + 5×4 neighbor features = 28 total
    2. Action space reduced to 5 (matches described neighbors)
    3. Reward shaping unchanged (already well-tuned)
    4. Everything else identical
    """

    metadata = {'render_modes': ['human']}

    # How many neighbors to describe in the state
    N_NEIGHBOR_SLOTS = 5
    # Features per neighbor: [dist_to_dest, battery, connectivity, is_valid]
    FEATURES_PER_NEIGHBOR = 4
    # Global features count
    N_GLOBAL_FEATURES = 8
    # Total state size
    STATE_SIZE = N_GLOBAL_FEATURES + N_NEIGHBOR_SLOTS * FEATURES_PER_NEIGHBOR  # 28

    def __init__(self, network: MeshNetwork, max_hops: int = 15):
        super(StableMeshRoutingEnv, self).__init__()

        self.network = network
        self.max_hops = max_hops
        self.current_node_id = None
        self.current_message = None
        self.step_count = 0
        self.initial_distance = 0

        # Statistics
        self.total_energy_used = 0
        self.messages_delivered = 0
        self.messages_dropped = 0
        self.total_episodes = 0

        # Enhanced state space: 28 features
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.STATE_SIZE,),
            dtype=np.float32
        )

        # Action space matches N_NEIGHBOR_SLOTS
        self.max_neighbors = self.N_NEIGHBOR_SLOTS
        self.action_space = spaces.Discrete(self.max_neighbors)

        self.max_network_size = 1000

    def _get_sorted_neighbors(self, node_id: str) -> List[str]:
        """Return active neighbors sorted by distance to destination (ascending)."""
        node = self.network.nodes[node_id]
        dest_node = self.network.nodes[self.current_message.destination_id]

        active_neighbors = [
            nid for nid in node.neighbors
            if self.network.nodes[nid].is_active
        ]

        active_neighbors.sort(
            key=lambda nid: self.network.nodes[nid].distance_to(dest_node)
        )

        return active_neighbors

    def _get_state(self, node_id: str, message: Message) -> np.ndarray:
        """
        28-feature state:
          [8 global features] + [5 neighbors × 4 features]

        The neighbor features directly correspond to the action choices,
        so the agent can now make an informed decision for each action.
        """
        node = self.network.nodes[node_id]
        dest_node = self.network.nodes[message.destination_id]

        # ── Global features (same 8 as before) ──────────────────────────────
        active_neighbors_nodes = [
            self.network.nodes[nid] for nid in node.neighbors
            if self.network.nodes[nid].is_active
        ]

        current_battery    = node.battery / 100.0
        num_neighbors      = min(len(active_neighbors_nodes) / 8.0, 1.0)
        hops_ratio         = min(message.hops / self.max_hops, 1.0)
        current_distance   = node.distance_to(dest_node)
        normalized_dist    = min(current_distance / self.max_network_size, 1.0)

        if self.initial_distance > 0:
            progress = max(0.0, min(1.0,
                (self.initial_distance - current_distance) / self.initial_distance))
        else:
            progress = 0.0

        at_dest = 1.0 if node_id == message.destination_id else 0.0

        if active_neighbors_nodes:
            best_dist      = min(n.distance_to(dest_node) for n in active_neighbors_nodes)
            best_neighbor_dist = min(best_dist / self.max_network_size, 1.0)
            avg_battery    = float(np.mean([n.battery / 100.0 for n in active_neighbors_nodes]))
        else:
            best_neighbor_dist = 1.0
            avg_battery        = 0.0

        global_features = np.array([
            current_battery,
            num_neighbors,
            hops_ratio,
            normalized_dist,
            progress,
            at_dest,
            best_neighbor_dist,
            avg_battery,
        ], dtype=np.float32)

        # ── Per-neighbor features ────────────────────────────────────────────
        # Get sorted neighbors (same order as action space)
        sorted_neighbors = self._get_sorted_neighbors(node_id)

        neighbor_features = np.zeros(
            self.N_NEIGHBOR_SLOTS * self.FEATURES_PER_NEIGHBOR,
            dtype=np.float32
        )

        for slot in range(self.N_NEIGHBOR_SLOTS):
            base = slot * self.FEATURES_PER_NEIGHBOR

            if slot < len(sorted_neighbors):
                nid    = sorted_neighbors[slot]
                nnode  = self.network.nodes[nid]

                # Feature 1: distance to destination (normalized)
                ndist  = nnode.distance_to(dest_node)
                neighbor_features[base + 0] = min(ndist / self.max_network_size, 1.0)

                # Feature 2: battery (normalized)
                neighbor_features[base + 1] = nnode.battery / 100.0

                # Feature 3: connectivity — how many active neighbors does
                #             this neighbor have? (normalized by 10)
                n_active_nbrs = sum(
                    1 for nn in nnode.neighbors
                    if self.network.nodes[nn].is_active
                )
                neighbor_features[base + 2] = min(n_active_nbrs / 10.0, 1.0)

                # Feature 4: slot is valid (neighbor exists)
                neighbor_features[base + 3] = 1.0
            # else: all zeros (slot is empty / invalid)

        return np.concatenate([global_features, neighbor_features])

    def _calculate_reward(self, old_distance: float, new_distance: float,
                          delivered: bool, dropped: bool,
                          energy_used: float, hops: int) -> float:
        """Reward shaping — unchanged from previous version."""
        if delivered:
            reward = 50.0
            efficiency = max(0, (self.max_hops - hops) / self.max_hops)
            reward += efficiency * 10.0
            self.messages_delivered += 1
            return float(np.clip(reward, -20.0, 60.0))

        if dropped:
            reward = -20.0
            self.messages_dropped += 1
            return float(np.clip(reward, -20.0, 60.0))

        distance_improvement = old_distance - new_distance

        if distance_improvement > 0:
            reward = 2.0
        elif distance_improvement < 0:
            reward = -1.0
        else:
            reward = -0.2

        reward -= energy_used * 0.05
        reward -= 0.05

        return float(np.clip(reward, -20.0, 60.0))

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        for node in self.network.nodes.values():
            node.is_active = True
            node.battery = 100.0
            node.message_queue.clear()

        self.network.update_all_neighbors()

        active_nodes = [n for n in self.network.get_active_nodes()
                        if len(n.neighbors) > 0]
        if len(active_nodes) < 2:
            raise RuntimeError("Network too small or disconnected after reset!")

        source_node = self.np_random.choice(active_nodes)
        dest_node   = self.np_random.choice(active_nodes)

        attempts = 0
        while (dest_node.id == source_node.id or
               source_node.distance_to(dest_node) < 400) and attempts < 50:
            dest_node = self.np_random.choice(active_nodes)
            attempts += 1

        self.current_message = Message(
            id=f"msg_{self.total_episodes}",
            source_id=source_node.id,
            destination_id=dest_node.id,
            content="Training message",
            timestamp=0.0
        )

        self.current_node_id  = source_node.id
        self.step_count       = 0
        self.total_energy_used = 0
        self.initial_distance = source_node.distance_to(dest_node)
        self.total_episodes  += 1

        state = self._get_state(self.current_node_id, self.current_message)
        return state, {'initial_distance': self.initial_distance}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        current_node = self.network.nodes[self.current_node_id]
        dest_node    = self.network.nodes[self.current_message.destination_id]

        old_distance     = current_node.distance_to(dest_node)
        sorted_neighbors = self._get_sorted_neighbors(self.current_node_id)

        if not sorted_neighbors:
            reward = self._calculate_reward(
                old_distance, old_distance,
                delivered=False, dropped=True,
                energy_used=0, hops=self.current_message.hops
            )
            state = self._get_state(self.current_node_id, self.current_message)
            return state, reward, True, False, {
                'result': 'dropped_isolated',
                'hops': self.current_message.hops,
                'energy_used': self.total_energy_used
            }

        # Clamp action to valid range
        neighbor_idx  = min(action, len(sorted_neighbors) - 1)
        next_node_id  = sorted_neighbors[neighbor_idx]
        next_node     = self.network.nodes[next_node_id]

        distance      = current_node.distance_to(next_node)
        energy_cost   = 0.1 + (distance / 150.0) * 0.05
        current_node.consume_energy(energy_cost)
        self.total_energy_used += energy_cost

        self.current_message.add_hop(next_node_id)
        self.step_count      += 1
        self.current_node_id  = next_node_id

        new_distance = next_node.distance_to(dest_node)
        delivered    = (next_node_id == self.current_message.destination_id)
        dropped      = (self.current_message.hops >= self.max_hops)

        reward = self._calculate_reward(
            old_distance, new_distance,
            delivered=delivered, dropped=dropped,
            energy_used=energy_cost,
            hops=self.current_message.hops
        )

        state = self._get_state(self.current_node_id, self.current_message)

        info = {
            'result': 'delivered' if delivered else ('dropped' if dropped else 'routing'),
            'hops': self.current_message.hops,
            'energy_used': self.total_energy_used
        }

        return state, reward, delivered or dropped, False, info

    def get_statistics(self) -> Dict:
        total = self.messages_delivered + self.messages_dropped
        return {
            'total_messages': total,
            'delivered': self.messages_delivered,
            'dropped': self.messages_dropped,
            'delivery_rate': self.messages_delivered / total if total > 0 else 0,
        }