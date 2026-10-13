import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Any, Dict, Tuple, Optional

import gymnasium as gym
from gymnasium import spaces


class TrafficLightEnv(gym.Env):
    """
    A simple traffic intersection environment for reinforcement learning.

    The intersection consists of four incoming lanes (North, South, East, West).
    Each lane has a queue of vehicles.  The agent selects a traffic light phase:

    * ``0`` – Green for North‑South (vehicles on N and S can depart)
    * ``1`` – Green for East‑West (vehicles on E and W can depart)

    At each step a random number of vehicles arrive on each lane and a fixed
    number of vehicles can depart from the lanes that have a green signal.
    The goal is to minimise the total queue length (or equivalently the total
    waiting time).

    Observation
    -----------
    ``observation``: ``Box(low=0, high=inf, shape=(4,), dtype=np.float32)``
        Current queue lengths for the four lanes in the order
        ``[North, South, East, West]``.

    Action
    ------
    ``action``: ``Discrete(2)``
        ``0`` – North‑South green, ``1`` – East‑West green.

    Reward
    ------
    ``-sum(queue_lengths)`` – negative total number of vehicles waiting.
    This encourages the agent to keep queues short.

    Episode termination
    --------------------
    The episode terminates after a fixed number of steps (``max_steps``).

    Rendering
    ---------
    The environment can be rendered with ``mode='human'`` using Matplotlib.
    Four vertical bars represent the queue lengths of each lane.
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        max_steps: int = 200,
        arrival_rate: float = 0.5,
        departure_rate: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialise the traffic light environment.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of steps per episode. Defaults to 200.
        arrival_rate : float, optional
            Expected number of arriving vehicles per lane per step
            (Poisson distributed). Defaults to 0.5.
        departure_rate : int, optional
            Maximum number of vehicles that can depart from a lane when its
            signal is green. Defaults to 2.
        seed : int, optional
            Random seed for reproducibility.
        """
        super().__init__()

        self.max_steps = max_steps
        self.arrival_rate = arrival_rate
        self.departure_rate = departure_rate

        # Observation: queue lengths for N, S, E, W (continuous, non‑negative)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        # Action: 0 = NS green, 1 = EW green
        self.action_space = spaces.Discrete(2)

        # Internal state
        self.queues: np.ndarray = np.zeros(4, dtype=np.int32)  # N, S, E, W
        self.current_step: int = 0
        self.current_phase: int = 0  # start with NS green

        # Random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Rendering helpers
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None

    def seed(self, seed: Optional[int] = None) -> list[int]:
        """
        Set the seed for this environment's random number generator.

        Parameters
        ----------
        seed : int, optional
            Seed value.

        Returns
        -------
        list[int]
            List containing the seed used.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator.
        options : dict, optional
            Additional information (unused).

        Returns
        -------
        observation : np.ndarray
            Initial observation (queue lengths).
        info : dict
            Additional information (empty).
        """
        if seed is not None:
            self.seed(seed)

        self.queues = np.zeros(4, dtype=np.int32)
        self.current_step = 0
        self.current_phase = 0  # start with NS green

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step of the environment's dynamics.

        Parameters
        ----------
        action : int
            Chosen traffic light phase (0 for NS green, 1 for EW green).

        Returns
        -------
        observation : np.ndarray
            Updated queue lengths.
        reward : float
            Negative total queue length.
        terminated : bool
            ``True`` if ``max_steps`` have been reached.
        truncated : bool
            Always ``False`` (no external truncation).
        info : dict
            Additional diagnostic information.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0 or 1.")

        self.current_phase = int(action)

        # 1. Vehicle arrivals (Poisson process per lane)
        arrivals = self.np_random.poisson(self.arrival_rate, size=4).astype(np.int32)
        self.queues += arrivals

        # 2. Vehicle departures from lanes with green signal
        if self.current_phase == 0:  # NS green
            green_lanes = [0, 1]  # indices for N and S
        else:  # EW green
            green_lanes = [2, 3]  # indices for E and W

        for lane in green_lanes:
            depart = min(self.queues[lane], self.departure_rate)
            self.queues[lane] -= depart

        # 3. Compute reward (negative total queue length)
        total_queue = float(self.queues.sum())
        reward = -total_queue

        # 4. Increment step counter and check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # no external truncation logic

        info = {
            "step": self.current_step,
            "queues": self.queues.copy(),
            "total_queue": total_queue,
            "action": self.current_phase,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Return the current observation as a float32 array."""
        return self.queues.astype(np.float32)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str, optional
            Rendering mode. Only ``'human'`` is supported.

        Returns
        -------
        img : np.ndarray or None
            If ``mode`` is ``'rgb_array'`` an image array would be returned.
            For ``'human'`` this method displays a Matplotlib figure and returns ``None``.
        """
        if mode != "human":
            raise NotImplementedError("Only 'human' render mode is implemented.")

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(4, 4))
            self._ax.set_title("Traffic Light Queues")
            self._ax.set_xlabel("Lane")
            self._ax.set_ylabel("Number of Vehicles")
            self._ax.set_xticks([0, 1, 2, 3])
            self._ax.set_xticklabels(["N", "S", "E", "W"])
            self._bars = self._ax.bar([0, 1, 2, 3], self.queues, color="C0")
            plt.ion()
            plt.show()
        else:
            for bar, height in zip(self._bars, self.queues):
                bar.set_height(height)

        # Add a visual cue for the current phase
        # Clear previous rectangle if present
        for artist in self._ax.patches:
            if isinstance(artist, Rectangle) and artist.get_facecolor() == (0.9, 0.9, 0.9, 1.0):
                artist.remove()
        if self.current_phase == 0:
            # Highlight N and S lanes
            self._ax.add_patch(
                Rectangle((-0.5, 0), 2, max(self.queues.max(), 1), color="yellow", alpha=0.2)
            )
        else:
            # Highlight E and W lanes
            self._ax.add_patch(
                Rectangle((1.5, 0), 2, max(self.queues.max(), 1), color="yellow", alpha=0.2)
            )

        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        return None

    def close(self) -> None:
        """Close the rendering window and clean up."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
