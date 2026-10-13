import random
import math
from collections import deque, namedtuple
from typing import Deque, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes
    ----------
    capacity: int
        Maximum number of transitions to store.
    buffer: Deque[Transition]
        Container that holds the transitions.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        """
        Parameters
        ----------
        capacity: int
            Maximum number of transitions that can be stored.
        """
        self.capacity: int = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a new transition to the buffer."""
        self.buffer.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Randomly sample a batch of transitions.

        Returns
        -------
        Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        # Convert each field to a torch tensor
        states = torch.from_numpy(
            np.stack([t.state for t in batch])
        ).float()
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long
        ).unsqueeze(1)
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float
        ).unsqueeze(1)
        next_states = torch.from_numpy(
            np.stack([t.next_state for t in batch])
        ).float()
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones


class DQNNetwork(nn.Module):
    """Simple fully‑connected Q‑network.

    The architecture can be customised by passing a list of hidden layer sizes.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Optional[List[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        input_dim: int
            Dimension of the state vector.
        output_dim: int
            Number of possible actions.
        hidden_layers: list[int] | None
            Sizes of hidden layers. If ``None`` a default of [128, 128] is used.
        """
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 128]

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class DQNAgent:
    """Deep Q‑Network agent for adaptive traffic‑light control.

    The agent interacts with an environment that follows the Gymnasium API.
    It uses an epsilon‑greedy policy for exploration and a target network
    to stabilize learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Optional[List[int]] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 500,
        target_update_freq: int = 1000,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Parameters
        ----------
        state_dim: int
            Dimensionality of the observation space.
        action_dim: int
            Number of discrete actions (traffic‑light phases).
        hidden_layers: list[int] | None
            Architecture of the Q‑network.
        lr: float
            Learning rate for the optimizer.
        gamma: float
            Discount factor.
        buffer_capacity: int
            Maximum size of the replay buffer.
        batch_size: int
            Number of samples per learning step.
        epsilon_start: float
            Initial exploration probability.
        epsilon_end: float
            Minimum exploration probability.
        epsilon_decay: float
            Decay rate controlling how fast epsilon shrinks.
        target_update_freq: int
            Number of steps between target‑network updates.
        device: torch.device | None
            Torch device; if ``None`` automatically selects CUDA if available.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Networks
        self.policy_net = DQNNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.target_net = DQNNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        # Internal step counter (used for epsilon schedule & target updates)
        self._steps_done = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Return an action for ``state`` using epsilon‑greedy policy.

        Parameters
        ----------
        state: np.ndarray
            Current observation.
        evaluate: bool
            If ``True`` the agent acts greedily (epsilon = 0).

        Returns
        -------
        int
            Chosen action index.
        """
        epsilon = (
            0.0
            if evaluate
            else self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * math.exp(-1.0 * self._steps_done / self.epsilon_decay)
        )
        self._steps_done += 1

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        # Convert state to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def _compute_td_error(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the temporal‑difference error for a batch."""
        # Current Q estimates
        q_values = self.policy_net(states).gather(1, actions)

        # Target Q values using the target network
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_max

        return q_values - target_q

    def learn(self) -> Optional[float]:
        """Sample a batch, perform a gradient step and update target net if needed.

        Returns
        -------
        loss: float | None
            The scalar loss value for the performed update, or ``None`` if the
            replay buffer does not contain enough samples.
        """
        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute loss (Huber loss is robust to outliers)
        td_error = self._compute_td_error(
            states, actions, rewards, next_states, dones
        )
        loss = nn.functional.smooth_l1_loss(
            td_error, torch.zeros_like(td_error)
        )

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to improve stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Periodically update the target network
        if self._steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self) -> None:
        """Hard update: copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str) -> None:
        """Save model parameters and optimizer state."""
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self._steps_done,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load model and optimizer state from ``filepath``."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._steps_done = checkpoint.get("steps_done", 0)
        # Restore epsilon schedule parameters if they were saved
        self.epsilon_start = checkpoint.get("epsilon_start", self.epsilon_start)
        self.epsilon_end = checkpoint.get("epsilon_end", self.epsilon_end)
        self.epsilon_decay = checkpoint.get(
            "epsilon_decay", self.epsilon_decay
        )
        # Ensure networks are on the correct device after loading
        self.policy_net.to(self.device)
        self.target_net.to(self.device)