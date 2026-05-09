from setuptools import setup, find_packages

setup(
    name="meshnet-ai",
    version="1.0.0",
    description="AI-powered mesh network disaster recovery for Tunisia using DQN & PPO",
    author="Haithem Nasr",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "stable-baselines3>=2.0.0",
        "gymnasium>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "networkx>=3.0",
        "dash>=2.14.0",
        "plotly>=5.18.0",
        "simpy>=4.0.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "meshnet-dashboard=dashboard:main",
            "meshnet-train=train_rl_agents:main",
            "meshnet-compare=comparison_test:main",
        ]
    },
)
