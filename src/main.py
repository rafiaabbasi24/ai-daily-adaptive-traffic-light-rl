import argparse
import logging
import random
from collections import deque, namedtuple
from pathlib import Path
from typing import Deque, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming the custom traffic light environment is defined in src.env
# The environment must follow the Gymnasium API (reset, step, render, etc.).
try:
    from src.env import TrafficLightEnv  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Could not import TrafficLightEnv. Ensure that src/env.py defines "
        "a Gymnasium-compatible environment named TrafficLightEnv."
    ) from exc

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --------------------------------------------------------------------------- #
# Replay Buffer
# --------------------------------------------------------------------------- #
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a new experience to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Randomly sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# Neural Network for the Q‑function
# --------------------------------------------------------------------------- #
class DQN(nn.Module):
    """Simple feed‑forward network for approximating Q‑values."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Agent encapsulating policy, training, and action selection
# --------------------------------------------------------------------------- #
class DQNAgent:
    """Deep Q‑Network agent with epsilon‑greedy exploration."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_capacity: int = 10000,
        target_update_freq: int = 10,
    ):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_capacity)

        self.steps_done = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon‑greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize_model(self) -> None:
        """Perform a single optimization step on a batch of transitions."""
        if len(self.replay_buffer) < self.batch_size:
            return

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

        # Huber loss
        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Periodically update the target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1

    def save(self, path: Path) -> None:
        """Save the policy network parameters."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: Path) -> None:
        """Load policy network parameters."""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# --------------------------------------------------------------------------- #
# Training & Evaluation utilities
# --------------------------------------------------------------------------- #
def linear_epsilon_decay(
    step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    """Linearly decay epsilon from start to end over decay_steps."""
    if step >= decay_steps:
        return end
    return start - (start - end) * (step / decay_steps)


def train(
    env: gym.Env,
    agent: DQNAgent,
    args: argparse.Namespace,
) -> Tuple[list, list]:
    """Run the training loop."""
    episode_rewards = []
    moving_avgs = []

    total_steps = 0
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for t in range(args.max_steps):
            epsilon = linear_epsilon_decay(
                total_steps,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay,
            )
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.optimize_model()

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        moving_avg = np.mean(episode_rewards[-100:])
        moving_avgs.append(moving_avg)

        logging.info(
            f"Episode {episode:4d} | Reward: {episode_reward:6.2f} | "
            f"Avg(100): {moving_avg:6.2f} | Epsilon: {epsilon:.3f}"
        )

        # Save checkpoint periodically
        if episode % args.checkpoint_interval == 0:
            checkpoint_path = args.save_dir / f"checkpoint_ep{episode}.pth"
            agent.save(checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

    return episode_rewards, moving_avgs


def evaluate(
    env: gym.Env,
    agent: DQNAgent,
    episodes: int = 10,
    render: bool = False,
) -> float:
    """Run evaluation episodes without exploration and return average reward."""
    total_reward = 0.0
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)  # greedy
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if render:
                env.render()
        total_reward += ep_reward
        logging.info(f"Eval Episode {ep + 1}/{episodes} | Reward: {ep_reward:.2f}")
    avg_reward = total_reward / episodes
    logging.info(f"Average evaluation reward over {episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def plot_learning_curve(
    rewards: list,
    moving_avg: list,
    save_path: Path,
) -> None:
    """Generate and save a plot of the learning curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Episode Reward", alpha=0.6)
    plt.plot(moving_avg, label="Moving Avg (100)", linewidth=2)
    plt.title("Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Learning curve saved to {save_path}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Traffic Light Control with Reinforcement Learning"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes (default: 500)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini‑batch size for optimization (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon for exploration (default: 1.0)",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon after decay (default: 0.05)",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=5000,
        help="Number of steps over which epsilon decays (default: 5000)",
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=100,
        help="Frequency (in steps) to update target network (default: 100)",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=10000,
        help="Capacity of the replay buffer (default: 10000)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save a checkpoint every N episodes (default: 50)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store model checkpoints and final model",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("reports/learning_curve.png"),
        help="Path to save the learning curve plot",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for final evaluation (default: 10)",
    )
    parser.add_argument(
        "--render-eval",
        action="store_true",
        help="Render the environment during evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set deterministic behavior where possible
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create environment
    env = TrafficLightEnv()
    assert isinstance(env, gym.Env), "TrafficLightEnv must inherit from gym.Env"

    # Extract dimensions
    sample_state, _ = env.reset()
    state_dim = np.asarray(sample_state).shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        raise ValueError("TrafficLightEnv must use a Discrete action space.")

    # Initialise agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_freq=args.target_update,
    )

    # Training
    rewards, moving_avg = train(env, agent, args)

    # Save final model
    final_model_path = args.save_dir / "dqn_traffic_light.pth"
    agent.save(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # Plot learning curve
    plot_learning_curve(rewards, moving_avg, args.plot_path)

    # Evaluation
    evaluate(env, agent, episodes=args.eval_episodes, render=args.render_eval)

    env.close()


if __name__ == "__main__":
    main()
