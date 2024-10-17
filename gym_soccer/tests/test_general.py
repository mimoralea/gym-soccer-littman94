import pytest
import numpy as np
from gym_soccer.envs.soccer_simultaneous_env import SoccerSimultaneousEnv

@pytest.mark.parametrize("width,height", [
    (5, 4),  # Minimum size, even height
    (6, 4),  # even height
    (7, 5),  # Odd height
    (9, 6),  # Even height
    (11, 7),  # Odd height
])
def test_initial_state_distribution(width, height):
    env = SoccerSimultaneousEnv(width=width, height=height)

    # Check that the total probability sums to 1
    total_prob = sum(prob for prob, _ in env.isd)
    assert abs(total_prob - 1.0) < 1e-6, f"Total probability should be 1, but is {total_prob}"

    # Check that all probabilities are equal
    first_prob = env.isd[0][0]
    assert all(abs(prob - first_prob) < 1e-6 for prob, _ in env.isd), "All probabilities should be equal"

    # Check starting positions
    for _, state in env.isd:
        row_a, col_a, row_b, col_b, possession = state

        # Check columns
        assert col_a == 2, f"Player A should start in column 2, but starts in column {col_a}"
        assert col_b == env.width - 3, f"Player B should start in column {env.width - 3}, but starts in column {col_b}"

        # Check rows
        if len(env.goal_rows) % 2 == 0:  # Even number of goal rows
            middle_index = len(env.goal_rows) // 2
            valid_rows = [env.goal_rows[middle_index - 1], env.goal_rows[middle_index]]
            assert row_a in valid_rows, f"Player A should start in row {valid_rows[0]} or {valid_rows[1]}, but starts in row {row_a}"
            assert row_b in valid_rows, f"Player B should start in row {valid_rows[0]} or {valid_rows[1]}, but starts in row {row_b}"
            assert row_a != row_b, f"Players should not start in the same row, but both start in row {row_a}"
        else:  # Odd number of goal rows
            middle_row = env.goal_rows[len(env.goal_rows) // 2]
            assert row_a == middle_row, f"Player A should start in middle row {middle_row}, but starts in row {row_a}"
            assert row_b == middle_row, f"Player B should start in middle row {middle_row}, but starts in row {row_b}"

        # Check possession
        assert possession in [0, 1], f"Possession should be 0 or 1, but is {possession}"

    # Check number of initial states
    if len(env.goal_rows) % 2 == 0:
        expected_states = 4  # Two row combinations, two possession states
    else:
        expected_states = 2  # One middle row, two possession states

    assert len(env.isd) == expected_states, f"Expected {expected_states} initial states, but got {len(env.isd)}"


@pytest.mark.parametrize("width,height", [
    (5, 4),  # Minimum size, even height
    (6, 4),  # even height
    (7, 5),  # Odd height
    (9, 6),  # Even height
    (11, 7),  # Odd height
])
def test_initial_state_sampling(width, height):
    env = SoccerSimultaneousEnv(width=width, height=height)
    n_samples = 10000
    state_counts = {}

    for _ in range(n_samples):
        env.reset()
        state_counts[env.state] = state_counts.get(env.state, 0) + 1

    total_states = len(state_counts)
    expected_prob = 1 / total_states
    expected_count = n_samples / total_states
    rtol = 0.1  # 10% relative tolerance

    for state, count in state_counts.items():
        row_a, col_a, row_b, col_b, possession = state

        # Check columns
        assert col_a == 2, f"Player A should start in column 2, but starts in column {col_a}"
        assert col_b == env.width - 3, f"Player B should start in column {env.width - 3}, but starts in column {col_b}"

        # Check rows
        if len(env.goal_rows) % 2 == 0:  # Even number of goal rows
            middle_index = len(env.goal_rows) // 2
            valid_rows = [env.goal_rows[middle_index - 1], env.goal_rows[middle_index]]
            assert row_a in valid_rows, f"Player A should start in row {valid_rows[0]} or {valid_rows[1]}, but starts in row {row_a}"
            assert row_b in valid_rows, f"Player B should start in row {valid_rows[0]} or {valid_rows[1]}, but starts in row {row_b}"
            assert row_a != row_b, f"Players should not start in the same row, but both start in row {row_a}"
        else:  # Odd number of goal rows
            middle_row = env.goal_rows[len(env.goal_rows) // 2]
            assert row_a == middle_row, f"Player A should start in middle row {middle_row}, but starts in row {row_a}"
            assert row_b == middle_row, f"Player B should start in middle row {middle_row}, but starts in row {row_b}"

        # Check possession
        assert possession in [0, 1], f"Possession should be 0 or 1, but is {possession}"

        # Check if the count is approximately equal to the expected count
        assert np.isclose(count, expected_count, rtol=rtol), \
            f"State {state} appeared {count} times, expected close to {expected_count}"

    # Check number of initial states
    if len(env.goal_rows) % 2 == 0:
        expected_states = 4  # Two row combinations, two possession states
    else:
        expected_states = 2  # One middle row, two possession states

    assert total_states == expected_states, f"Expected {expected_states} initial states, but got {total_states}"

    # Check that the empirical probabilities are close to the expected probability
    observed = np.array(list(state_counts.values()))
    empirical_probs = observed / n_samples
    assert np.allclose(empirical_probs, expected_prob, rtol=rtol), \
        f"Empirical probabilities {empirical_probs} not close to expected {expected_prob}"

    # Check for uniformity using coefficient of variation
    cv = np.std(observed) / np.mean(observed)
    assert cv < 0.05, f"Distribution not uniform enough. Coefficient of variation: {cv:.3f}"


def test_singleagent_a():
    from gym import spaces
    width = 5
    height = 4
    slip_prob = 0.2
    n_states = 761 # 4x5 field
    n_actions = 5
    random_policy = {}
    for s in range(n_states):
        random_policy[s] = np.random.randint(0, n_actions)

    env = SoccerSimultaneousEnv(width=width, height=height, slip_prob=slip_prob, player_a_policy=None, player_b_policy=random_policy)
    assert not env.multiagent, "Environment should not be multiagent, one policy was provided."

    # Check that the observation space is Dict
    assert isinstance(env.observation_space, spaces.Dict), "Observation space should be a dictionary."
    assert env.observation_space['player_a'].n == n_states, "Observation space should have the correct number of states."
    assert 'player_b' not in env.observation_space, "Observation space should not contain player_b."

    # Check that the action space is Dict
    assert isinstance(env.action_space, spaces.Dict), "Action space should be a dictionary."
    assert env.action_space['player_a'].n == n_actions, "Action space should have the correct number of actions."
    assert 'player_b' not in env.action_space, "Action space should not contain player_b."
    
    obs, info = env.reset()
    assert isinstance(obs, dict), "Observation should be a dictionary, single agent mode."
    assert 'player_a' in obs, "Observation should contain player_a."
    assert 'player_b' not in obs, "Observation should not contain player_b."
    assert 0 <= obs['player_a'] < n_states, "Observation should be a state index."
    assert isinstance(info, dict), "Info should be a dictionary."
    assert 'player_a' in info, "Info should contain player_a."
    assert 'player_b' not in info, "Info should not contain player_b."

    random_action = np.random.randint(0, n_actions)
    obs, reward, terminated, truncated, info = env.step({'player_a': random_action})
    assert isinstance(obs, dict), "Observation should be a dictionary, single agent mode."
    assert 'player_a' in obs, "Observation should contain player_a."
    assert 'player_b' not in obs, "Observation should not contain player_b."
    assert 0 <= obs['player_a'] < n_states, "Observation should be a state index."
    assert isinstance(reward, dict), "Reward should be a dictionary."
    assert 'player_a' in reward, "Reward should contain player_a."
    assert 'player_b' not in reward, "Reward should not contain player_b."
    assert isinstance(terminated, dict), "Terminated should be a dictionary."
    assert 'player_a' in terminated, "Terminated should contain player_a."
    assert 'player_b' not in terminated, "Terminated should not contain player_b."
    assert isinstance(truncated, dict), "Truncated should be a dictionary."
    assert 'player_a' in truncated, "Truncated should contain player_a."
    assert 'player_b' not in truncated, "Truncated should not contain player_b."
    assert isinstance(info, dict), "Info should be a dictionary."
    assert 'player_a' in info, "Info should contain player_a."
    assert 'player_b' not in info, "Info should not contain player_b."

def test_singleagent_b():
    from gym import spaces
    width = 5
    height = 4
    slip_prob = 0.2
    n_states = 761 # 4x5 field
    n_actions = 5
    random_policy = {}
    for s in range(n_states):
        random_policy[s] = np.random.randint(0, n_actions)

    env = SoccerSimultaneousEnv(width=width, height=height, slip_prob=slip_prob, player_a_policy=random_policy, player_b_policy=None)
    assert not env.multiagent, "Environment should not be multiagent, one policy was provided."

    # Check that the observation space is Dict
    assert isinstance(env.observation_space, spaces.Dict), "Observation space should be a dictionary."
    assert env.observation_space['player_b'].n == n_states, "Observation space should have the correct number of states."
    assert 'player_a' not in env.observation_space, "Observation space should not contain player_a."

    # Check that the action space is Dict
    assert isinstance(env.action_space, spaces.Dict), "Action space should be a dictionary."
    assert env.action_space['player_b'].n == n_actions, "Action space should have the correct number of actions."
    assert 'player_a' not in env.action_space, "Action space should not contain player_a."
    
    obs, info = env.reset()
    assert isinstance(obs, dict), "Observation should be a dictionary, single agent mode."
    assert 'player_b' in obs, "Observation should contain player_b."
    assert 'player_a' not in obs, "Observation should not contain player_a."
    assert 0 <= obs['player_b'] < n_states, "Observation should be a state index."
    assert isinstance(info, dict), "Info should be a dictionary."
    assert 'player_b' in info, "Info should contain player_b."
    assert 'player_a' not in info, "Info should not contain player_a."

    random_action = np.random.randint(0, n_actions)
    obs, reward, terminated, truncated, info = env.step({'player_b': random_action})
    assert isinstance(obs, dict), "Observation should be a dictionary, single agent mode."
    assert 'player_b' in obs, "Observation should contain player_b."
    assert 'player_a' not in obs, "Observation should not contain player_a."
    assert 0 <= obs['player_b'] < n_states, "Observation should be a state index."
    assert isinstance(reward, dict), "Reward should be a dictionary."
    assert 'player_b' in reward, "Reward should contain player_b."
    assert 'player_a' not in reward, "Reward should not contain player_a."
    assert isinstance(terminated, dict), "Terminated should be a dictionary."
    assert 'player_b' in terminated, "Terminated should contain player_b."
    assert 'player_a' not in terminated, "Terminated should not contain player_a."
    assert isinstance(truncated, dict), "Truncated should be a dictionary."
    assert 'player_b' in truncated, "Truncated should contain player_b."
    assert 'player_a' not in truncated, "Truncated should not contain player_a."
    assert isinstance(info, dict), "Info should be a dictionary."
    assert 'player_b' in info, "Info should contain player_b."
    assert 'player_a' not in info, "Info should not contain player_a."

def test_multiagent():
    from gym import spaces
    width = 5
    height = 4
    slip_prob = 0.2
    n_states = 761 # 4x5 field
    n_actions = 5

    env = SoccerSimultaneousEnv(width=width, height=height, slip_prob=slip_prob, player_a_policy=None, player_b_policy=None)
    assert env.multiagent, "Environment should be multiagent, no policies were provided."

    # Check that the observation space is Dict
    assert isinstance(env.observation_space, spaces.Dict), "Observation space should be a dictionary."
    assert env.observation_space['player_a'].n == n_states, "Observation space should have the correct number of states for player_a."
    assert env.observation_space['player_b'].n == n_states, "Observation space should have the correct number of states for player_b."

    # Check that the action space is Dict
    assert isinstance(env.action_space, spaces.Dict), "Action space should be a dictionary."
    assert env.action_space['player_a'].n == n_actions, "Action space should have the correct number of actions for player_a."
    assert env.action_space['player_b'].n == n_actions, "Action space should have the correct number of actions for player_b."

    obs, info = env.reset()
    assert isinstance(obs, dict), "Observation should be a dictionary, multiagent mode."
    assert 'player_a' in obs and 'player_b' in obs, "Observation should contain both player_a and player_b."
    assert 0 <= obs['player_a'] < n_states and 0 <= obs['player_b'] < n_states, "Observations should be state indices."
    assert isinstance(info, dict), "Info should be a dictionary."
    assert 'player_a' in info and 'player_b' in info, "Info should contain both player_a and player_b."

    random_action_a = np.random.randint(0, n_actions)
    random_action_b = np.random.randint(0, n_actions)
    obs, reward, terminated, truncated, info = env.step({'player_a': random_action_a, 'player_b': random_action_b})
    assert isinstance(obs, dict), "Observation should be a dictionary, multiagent mode."
    assert 'player_a' in obs and 'player_b' in obs, "Observation should contain both player_a and player_b."
    assert 0 <= obs['player_a'] < n_states and 0 <= obs['player_b'] < n_states, "Observations should be state indices."
    assert isinstance(reward, dict), "Reward should be a dictionary."
    assert 'player_a' in reward and 'player_b' in reward, "Reward should contain both player_a and player_b."
    assert isinstance(reward['player_a'], float) and isinstance(reward['player_b'], float), "Rewards should be floats."
    assert isinstance(terminated, dict) and isinstance(terminated['player_a'], bool) and isinstance(terminated['player_b'], bool), "Terminated should be a dictionary with boolean values."
    assert isinstance(truncated, dict) and isinstance(truncated['player_a'], bool) and isinstance(truncated['player_b'], bool), "Truncated should be a dictionary with boolean values."
    assert isinstance(info, dict) and 'player_a' in info and 'player_b' in info, "Info should be a dictionary with both player_a and player_b."
