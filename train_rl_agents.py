"""
Train RL agents (DQN and PPO) for mesh network routing
ENHANCED VERSION: Larger state (28 features), 500k timesteps, bigger networks
"""

import os
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from src.network.node import MeshNode
from src.network.network import MeshNetwork
from src.ai.stable_rl_env import StableMeshRoutingEnv
from src.scenarios.tunisia_map import TUNISIAN_CITIES
import random

MAX_HOPS = 15


def create_tunisia_network(num_nodes_per_city: int = 3,
                           transmission_range: float = 260):
    network = MeshNetwork()
    cities_to_use = dict(list(TUNISIAN_CITIES.items())[:10])

    for city_name, city in cities_to_use.items():
        for i in range(num_nodes_per_city):
            offset_x = random.uniform(-15, 15)
            offset_y = random.uniform(-15, 15)
            node = MeshNode(
                node_id=f"{city_name}_{i}",
                x=city.x + offset_x,
                y=city.y + offset_y,
                transmission_range=transmission_range,
                initial_battery=100.0
            )
            network.add_node(node)
    return network


def make_env(transmission_range=260):
    def _init():
        network = create_tunisia_network(transmission_range=transmission_range)
        env = StableMeshRoutingEnv(network, max_hops=MAX_HOPS)
        env = Monitor(env)
        return env
    return _init


def train_dqn(total_timesteps: int = 500_000, save_dir: str = "models/dqn"):
    print("\n" + "=" * 70)
    print("TRAINING DQN — Enhanced 28-feature state, 500k steps")
    print("=" * 70)
    os.makedirs(save_dir, exist_ok=True)

    env = DummyVecEnv([make_env()])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=f"{save_dir}/tensorboard/"
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000, save_path=save_dir, name_prefix="dqn_mesh"
    )

    print(f"State size: {model.observation_space.shape[0]} features")
    print(f"Action size: {model.action_space.n} actions")
    print(f"Training for {total_timesteps:,} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        log_interval=20,
        progress_bar=True
    )

    model.save(f"{save_dir}/dqn_final")
    print(f"\n✓ DQN saved to: {save_dir}/dqn_final.zip")
    return model


def train_ppo(total_timesteps: int = 500_000, save_dir: str = "models/ppo"):
    print("\n" + "=" * 70)
    print("TRAINING PPO — Enhanced 28-feature state, 500k steps")
    print("=" * 70)
    os.makedirs(save_dir, exist_ok=True)

    env = DummyVecEnv([make_env()])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,          
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        tensorboard_log=f"{save_dir}/tensorboard/"
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000, save_path=save_dir, name_prefix="ppo_mesh"
    )

    print(f"State size: {model.observation_space.shape[0]} features")
    print(f"Action size: {model.action_space.n} actions")
    print(f"Training for {total_timesteps:,} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        log_interval=20,
        progress_bar=True
    )

    model.save(f"{save_dir}/ppo_final")
    print(f"\n✓ PPO saved to: {save_dir}/ppo_final.zip")
    return model


def evaluate_model(model, network, n_episodes: int = 100) -> dict:
    env = StableMeshRoutingEnv(network, max_hops=MAX_HOPS)
    successes, total_hops, total_energy = 0, [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info['result'] == 'delivered':
            successes += 1
            total_hops.append(info['hops'])
            total_energy.append(info['energy_used'])

    return {
        'success_rate':  successes / n_episodes,
        'avg_hops':      np.mean(total_hops)   if total_hops   else 0,
        'avg_energy':    np.mean(total_energy) if total_energy else 0,
    }


def main():
    print("=" * 70)
    print("MESH NETWORK RL TRAINING — ENHANCED STATE SPACE")
    print(f"State: 28 features | Actions: 5 | max_hops: {MAX_HOPS}")
    print(f"Timesteps: 500,000 per agent")
    print("=" * 70)

    dqn_model = train_dqn(total_timesteps=500_000)
    ppo_model = train_ppo(total_timesteps=500_000)

    print("\n" + "=" * 70)
    print("QUICK EVALUATION (100 episodes)")
    print("=" * 70)

    test_net = create_tunisia_network()

    print("\n📊 DQN:")
    dqn_res = evaluate_model(dqn_model, test_net)
    for k, v in dqn_res.items():
        print(f"  {k}: {v:.3f}")

    print("\n📊 PPO:")
    ppo_res = evaluate_model(ppo_model, test_net)
    for k, v in ppo_res.items():
        print(f"  {k}: {v:.3f}")

    print("\n✅ Training complete! Run comparison_test.py to benchmark.")


if __name__ == "__main__":
    main()