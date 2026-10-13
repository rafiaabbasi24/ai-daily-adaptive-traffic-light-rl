import pytest
import numpy as np
import gymnasium as gym

# Adjust the import path according to the actual project structure.
# The tests assume that the environment is registered with Gymnasium under the ID
# "AdaptiveTrafficLight-v0". If the environment is provided as a class, replace the
# gym.make call with the appropriate import, e.g.:
# from adaptive_traffic_light.env import TrafficLightEnv
# env = TrafficLightEnv()
ENV_ID = "AdaptiveTrafficLight-v0"


def create_env():
    """
    Helper to create the environment. Tries to use gymnasium.make first;
    if the environment is not registered, attempts to import a class named
    `TrafficLightEnv` from the project's package.
    """
    try:
        env = gym.make(ENV_ID, render_mode=None)
        # gym.make returns a tuple (env, info) in newer versions; handle both.
        if isinstance(env, tuple):
            env = env[0]
        return env
    except Exception as exc:
        # Fallback: try to import the class directly.
        try:
            from adaptive_traffic_light.env import TrafficLightEnv  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"Unable to create the environment. Tried gym.make('{ENV_ID}') "
                f"and direct import, but both failed."
            ) from e
        return TrafficLightEnv()


@pytest.fixture(scope="function")
def env():
    """
    Pytest fixture that creates a fresh environment for each test and ensures
    proper cleanup.
    """
    env_instance = create_env()
    yield env_instance
    # Close the environment if it implements a close method.
    if hasattr(env_instance, "close"):
        env_instance.close()


def test_environment_creation(env):
    """
    Verify that the environment can be instantiated and that its spaces are defined.
    """
    assert env is not None, "Environment instance should not be None."
    assert hasattr(env, "observation_space"), "Environment must have an observation_space."
    assert hasattr(env, "action_space"), "Environment must have an action_space."

    # Check that spaces are instances of gymnasium.spaces.Space
    assert isinstance(env.observation_space, gym.spaces.Space), (
        "observation_space must be a gymnasium Space instance."
    )
    assert isinstance(env.action_space, gym.spaces.Space), (
        "action_space must be a gymnasium Space instance."
    )


def test_reset_returns_valid_observation(env):
    """
    Ensure that reset returns an observation that lies within the observation space.
    """
    observation, info = env.reset()
    # Observation should be a numpy array or a scalar compatible with the space.
    assert isinstance(observation, (np.ndarray, float, int, list, tuple)), (
        "Reset observation must be a numpy array, scalar, list, or tuple."
    )
    # Use the space's `contains` method for validation.
    assert env.observation_space.contains(observation), (
        "Reset observation is not within the defined observation_space."
    )
    # Info should be a dictionary (Gymnasium convention).
    assert isinstance(info, dict), "Reset info must be a dictionary."


def test_step_returns_valid_structure(env):
    """
    After resetting, take a random valid action and verify the step output.
    """
    observation, _ = env.reset()
    # Sample a valid action from the action space.
    action = env.action_space.sample()
    step_result = env.step(action)

    # Gymnasium step returns (obs, reward, terminated, truncated, info)
    assert isinstance(step_result, tuple) and len(step_result) == 5, (
        "Step must return a 5-tuple: (obs, reward, terminated, truncated, info)."
    )
    next_obs, reward, terminated, truncated, info = step_result

    # Observation validation
    assert isinstance(next_obs, (np.ndarray, float, int, list, tuple)), (
        "Step observation must be a numpy array, scalar, list, or tuple."
    )
    assert env.observation_space.contains(next_obs), (
        "Step observation is not within the defined observation_space."
    )

    # Reward validation
    assert isinstance(reward, (float, int, np.number)), "Reward must be a numeric type."
    # If the environment defines reward_range, ensure reward lies within it.
    if hasattr(env, "reward_range") and env.reward_range is not None:
        low, high = env.reward_range
        assert low <= reward <= high, (
            f"Reward {reward} out of expected range {env.reward_range}."
        )

    # Termination flags validation
    assert isinstance(terminated, bool), "terminated flag must be a bool."
    assert isinstance(truncated, bool), "truncated flag must be a bool."

    # Info validation
    assert isinstance(info, dict), "Info returned by step must be a dictionary."


def test_episode_termination(env):
    """
    Run an episode until termination (or a safety max step limit) and verify that
    the environment eventually signals termination or truncation.
    """
    max_steps = 500  # Safety cap to avoid infinite loops in buggy environments.
    observation, _ = env.reset()
    terminated = truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < max_steps:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # Basic sanity checks each step.
        assert isinstance(observation, (np.ndarray, float, int, list, tuple))
        assert env.observation_space.contains(observation)
        assert isinstance(reward, (float, int, np.number))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    assert terminated or truncated, (
        f"Episode did not terminate after {max_steps} steps; possible bug in the environment."
    )
    # Optional: ensure that after termination, further steps raise an error or keep returning terminal state.
    if terminated or truncated:
        with pytest.raises(Exception):
            # Many gym environments raise an error if step is called after termination.
            env.step(env.action_space.sample()) 

def test_action_space_sampling_consistency(env):
    """
    Verify that sampling from the action space always yields valid actions.
    """
    for _ in range(100):
        action = env.action_space.sample()
        assert env.action_space.contains(action), (
            "Sampled action is not within the defined action_space."
        )
    # Additionally, ensure that the action space is not empty.
    sample_action = env.action_space.sample()
    assert sample_action is not None, "Action space sampling returned None."

def test_observation_space_bounds(env):
    """
    For environments with bounded observation spaces (e.g., Box), ensure that
    the low and high bounds are finite and that reset respects them.
    """
    if isinstance(env.observation_space, gym.spaces.Box):
        low = env.observation_space.low
        high = env.observation_space.high
        # Ensure bounds are finite numbers.
        assert np.all(np.isfinite(low)), "Observation space low bounds must be finite."
        assert np.all(np.isfinite(high)), "Observation space high bounds must be finite."
        # Verify reset observation respects bounds.
        obs, _ = env.reset()
        assert np.all(obs >= low - 1e-6) and np.all(obs <= high + 1e-6), (
            "Reset observation violates observation space bounds."
        )