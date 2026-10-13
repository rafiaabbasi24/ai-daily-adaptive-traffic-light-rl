# Adaptive Traffic Light Control with Reinforcement Learning

## Project Overview
This repository implements an adaptive traffic signal controller based on **deep reinforcement learning**.  
The agent learns to minimize vehicle waiting time and queue length at a four‑way intersection by interacting with a simulated environment built with **Gymnasium**. The core learning algorithm uses **Proximal Policy Optimization (PPO)** implemented in **PyTorch**.

Key features:
- **Modular environment** (`gymnasium` compatible) that can be extended to more complex road networks.
- **PPO agent** with configurable neural network architecture, learning rate, and clipping parameters.
- **Comprehensive logging** (training curves, episode statistics) using `matplotlib`.
- **Reproducible experiments** thanks to deterministic seeding and configurable hyper‑parameters.
- **Visualization tools** for animated traffic flow and performance dashboards.

## Tech Stack
- Python ≥ 3.9
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

## Repository Structure
```
adaptive-traffic-light-rl/
├─ env/                     # Custom Gymnasium environment
│   ├─ traffic_env.py
│   └─ utils.py
├─ agent/                   # PPO implementation
│   ├─ ppo.py
│   └─ network.py
├─ scripts/                 # Entry points for training/evaluation
│   ├─ train.py
│   └─ evaluate.py
├─ plots/                   # Generated figures
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/adaptive-traffic-light-rl.git
cd adaptive-traffic-light-rl
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

The `requirements.txt` contains:
```
torch>=2.0.0
gymnasium>=0.28.0
numpy>=1.24.0
matplotlib>=3.7.0
```

### 4. Verify installation
```bash
python -c "import torch, gymnasium, numpy, matplotlib; print('All packages imported successfully')"
```

## Usage

### Training the agent
```bash
python scripts/train.py \
    --episodes 5000 \
    --max-steps-per-episode 200 \
    --learning-rate 3e-4 \
    --gamma 0.99 \
    --clip-epsilon 0.2 \
    --seed 42 \
    --log-dir runs/ppo_adaptive
```

- `--episodes`: total training episodes.
- `--max-steps-per-episode`: horizon for each episode.
- `--learning-rate`: PPO optimizer step size.
- `--gamma`: discount factor.
- `--clip-epsilon`: PPO clipping parameter.
- `--seed`: reproducibility seed.
- `--log-dir`: directory where TensorBoard logs and checkpoint files are saved.

Training progress can be monitored with TensorBoard:
```bash
tensorboard --logdir runs/ppo_adaptive
```

### Evaluating a trained policy
```bash
python scripts/evaluate.py \
    --checkpoint runs/ppo_adaptive/checkpoint_4000.pt \
    --episodes 20 \
    --render
```

- `--checkpoint`: path to a saved model.
- `--episodes`: number of evaluation runs.
- `--render`: if set, an animation of traffic flow is displayed using Matplotlib.

### Visualizing results
After training, the script automatically stores reward curves and queue‑length statistics under `plots/`. To generate a custom plot:
```python
import matplotlib.pyplot as plt
import numpy as np

log_file = "runs/ppo_adaptive/rewards.npy"
rewards = np.load(log_file)
plt.plot(rewards)
plt.title("Episode Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig("plots/reward_curve.png")
plt.show()
```

## Configuration
All hyper‑parameters can be overridden via command‑line arguments or by editing the default configuration in `scripts/train.py`. The environment itself can be customized (e.g., traffic arrival rates, number of lanes) by modifying `env/traffic_env.py`.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Ensure code style compliance with `flake8` and that all tests pass (`pytest`).
4. Submit a pull request with a clear description of changes.

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

## Citation
If you use this code in academic work, please cite:

```bibtex
@software{adaptive_traffic_rl,
  author       = {Your Name and Collaborators},
  title        = {Adaptive Traffic Light Control with Reinforcement Learning},
  year         = {2024},
  url          = {https://github.com/yourusername/adaptive-traffic-light-rl},
  note         = {Version 1.0.0}
}