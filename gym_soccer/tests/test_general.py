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
    isd = env._generate_isd()
    
    # Check that the total probability sums to 1
    total_prob = sum(prob for prob, _ in isd)
    assert abs(total_prob - 1.0) < 1e-6, f"Total probability should be 1, but is {total_prob}"
    
    # Check that all probabilities are equal
    first_prob = isd[0][0]
    assert all(abs(prob - first_prob) < 1e-6 for prob, _ in isd), "All probabilities should be equal"
    
    # Check starting positions
    for _, state in isd:
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
    
    assert len(isd) == expected_states, f"Expected {expected_states} initial states, but got {len(isd)}"


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
        state = tuple(env.state.values())
        state_counts[state] = state_counts.get(state, 0) + 1

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
