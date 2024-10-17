import pytest
import numpy as np
from gym_soccer.envs import SoccerSimultaneousEnv

@pytest.fixture
def env():
    return SoccerSimultaneousEnv(width=5, height=4, slip_prob=0.0)

@pytest.fixture(autouse=True)
def reset_env(env):
    env.reset()
    yield

def test_initialization(env):
    env.reset()
    assert env.width == 7  # 5 + 2 for goal columns
    assert env.height == 4
    assert env.slip_prob == 0.0
    assert env.action_space['player_a'].n == 5
    assert env.action_space['player_b'].n == 5

def test_reset(env):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert 'player_a' in obs and 'player_b' in obs
    assert isinstance(info, dict)
    assert 'player_a' in info and 'player_b' in info

def test_step(env):
    env.reset()
    action = {'player_a': env.NOOP, 'player_b': env.NOOP}
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, dict)
    assert isinstance(terminated, dict)
    assert isinstance(truncated, dict)
    assert isinstance(info, dict)

def test_scoring(env):
    def run_scoring_test(initial_state, action_a, action_b):
        env.reset()
        env.state = initial_state
        action = {'player_a': action_a, 'player_b': action_b}
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated['player_a'] and terminated['player_b'], "Game should end"
        assert abs(reward['player_a']) == 1 and abs(reward['player_b']) == 1, "Both players must receive a reward/penalty for a goal"

    # Test Player A scoring
    run_scoring_test((1, 5, 3, 1, 0), env.EAST, env.NOOP)

    # Test Player B scoring
    run_scoring_test((3, 5, 1, 1, 1), env.NOOP, env.WEST)

def test_own_goals(env):
    # Test Player A scoring an own goal (row 1)
    env.state = (1, 1, 3, 5, 0)  # Player A with ball, near own goal
    obs, reward, done, truncated, info = env.step({'player_a': env.WEST, 'player_b': env.NOOP})
    assert done['player_a'] and done['player_b']
    assert reward['player_a'] == -1
    assert reward['player_b'] == 1

    # Test Player A scoring an own goal (row 2)
    env.reset()
    env.state = (2, 1, 3, 5, 0)  # Player A with ball, near own goal
    obs, reward, done, truncated, info = env.step({'player_a': env.WEST, 'player_b': env.NOOP})
    assert done['player_a'] and done['player_b']
    assert reward['player_a'] == -1
    assert reward['player_b'] == 1

    # Test Player B scoring an own goal (row 1)
    env.reset()
    env.state = (3, 1, 1, 5, 1)  # Player B with ball, near own goal
    obs, reward, done, truncated, info = env.step({'player_a': env.NOOP, 'player_b': env.EAST})
    assert done['player_a'] and done['player_b']
    assert reward['player_a'] == 1
    assert reward['player_b'] == -1

    # Test Player B scoring an own goal (row 2)
    env.reset()
    env.state = (3, 1, 2, 5, 1)  # Player B with ball, near own goal
    obs, reward, done, truncated, info = env.step({'player_a': env.NOOP, 'player_b': env.EAST})
    assert done['player_a'] and done['player_b']
    assert reward['player_a'] == 1
    assert reward['player_b'] == -1

def test_both_players_moving_collision(env):
    # Test collision when both players are moving
    env.reset()
    env.state = (1, 2, 1, 3, 0)  # Player A has possession
    action = {'player_a': env.EAST, 'player_b': env.WEST}
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.state[1] == 2 and env.state[3] == 3, "Players should bounce back to original positions"
    assert env.state[4] in [0, 1], "Possession should be randomly assigned"

    env.reset()
    env.state = (1, 2, 1, 3, 1)  # Player B has possession
    action = {'player_a': env.EAST, 'player_b': env.WEST}
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.state[1] == 2 and env.state[3] == 3, "Players should bounce back to original positions"
    assert env.state[4] in [0, 1], "Possession should be randomly assigned"

def test_one_player_standing_collision(env):
    # Test collision when one player is standing still
    env.reset()
    env.state = (1, 2, 1, 3, 0)  # Player A has possession
    action = {'player_a': env.EAST, 'player_b': env.NOOP}
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.state[1] == 2 and env.state[3] == 3, "Players should remain in original positions"
    assert env.state[4] in [0, 1], "Possession should be randomly assigned"

    env.reset()
    env.state = (1, 2, 1, 3, 1)  # Player B has possession
    action = {'player_a': env.NOOP, 'player_b': env.WEST}
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.state[1] == 2 and env.state[3] == 3, "Players should remain in original positions"
    assert env.state[4] in [0, 1], "Possession should be randomly assigned"

def test_move_to_same_cell_collision(env):
    def run_move_to_same_cell_collision_test(initial_state, action_a, action_b, iterations=1000):
        move_success_counts = {'A': 0, 'B': 0}
        possession_switch_count = 0
        initial_possession = initial_state[4]

        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            action = {'player_a': action_a, 'player_b': action_b}
            obs, reward, terminated, truncated, info = env.step(action)

            if env.state[0] != initial_state[0] or env.state[1] != initial_state[1]:
                move_success_counts['A'] += 1
            elif env.state[2] != initial_state[2] or env.state[3] != initial_state[3]:
                move_success_counts['B'] += 1

            if env.state[4] != initial_possession:
                possession_switch_count += 1

        for player, count in move_success_counts.items():
            success_ratio = count / iterations
            assert 0.45 <= success_ratio <= 0.55, f"Move success ratio for Player {player}: {success_ratio:.2f}, expected close to 0.5"

        possession_switch_ratio = possession_switch_count / iterations
        assert 0.45 <= possession_switch_ratio <= 0.55, f"Possession switch ratio: {possession_switch_ratio:.2f}, expected close to 0.5"

    # Diagonal movements
    run_move_to_same_cell_collision_test((1, 1, 2, 2, 0), env.EAST, env.NORTH)    # A: right, B: up
    run_move_to_same_cell_collision_test((1, 1, 2, 2, 1), env.EAST, env.NORTH)    # Same, but B has initial possession
    run_move_to_same_cell_collision_test((1, 2, 2, 1, 0), env.WEST, env.NORTH)     # A: left, B: up
    run_move_to_same_cell_collision_test((1, 2, 2, 1, 1), env.WEST, env.NORTH)     # Same, but B has initial possession
    run_move_to_same_cell_collision_test((2, 1, 1, 2, 0), env.EAST, env.SOUTH)  # A: right, B: down
    run_move_to_same_cell_collision_test((2, 1, 1, 2, 1), env.EAST, env.SOUTH)  # Same, but B has initial possession
    run_move_to_same_cell_collision_test((2, 2, 1, 1, 0), env.WEST, env.SOUTH)   # A: left, B: down
    run_move_to_same_cell_collision_test((2, 2, 1, 1, 1), env.WEST, env.SOUTH)   # Same, but B has initial possession

    # Horizontal movements
    run_move_to_same_cell_collision_test((1, 1, 1, 3, 0), env.EAST, env.WEST)  # A: right, B: left
    run_move_to_same_cell_collision_test((1, 1, 1, 3, 1), env.EAST, env.WEST)  # Same, but B has initial possession
    run_move_to_same_cell_collision_test((1, 3, 1, 1, 0), env.WEST, env.EAST)  # A: left, B: right
    run_move_to_same_cell_collision_test((1, 3, 1, 1, 1), env.WEST, env.EAST)  # Same, but B has initial possession

    # Vertical movements
    run_move_to_same_cell_collision_test((1, 1, 3, 1, 0), env.SOUTH, env.NORTH)     # A: down, B: up
    run_move_to_same_cell_collision_test((1, 1, 3, 1, 1), env.SOUTH, env.NORTH)     # Same, but B has initial possession
    run_move_to_same_cell_collision_test((3, 1, 1, 1, 0), env.NORTH, env.SOUTH)     # A: up, B: down
    run_move_to_same_cell_collision_test((3, 1, 1, 1, 1), env.NORTH, env.SOUTH)     # Same, but B has initial possession

def test_all_edges(env):
    # Test Player A at top edge, B at right edge
    # Case 1: A has possession, A moves UP, B moves RIGHT
    initial_state = (0, 1, 3, 5, 0)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.NORTH, 'player_b': env.EAST})
    assert env.state == initial_state, "State should not change when players attempt to move out of bounds"

    # Case 2: B has possession, A moves UP, B moves RIGHT
    initial_state = (0, 1, 3, 5, 1)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.NORTH, 'player_b': env.EAST})
    assert env.state == initial_state, "State should not change when players attempt to move out of bounds"

    # Case 3: A has possession, A moves LEFT, B moves RIGHT
    initial_state = (0, 1, 3, 5, 0)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.WEST, 'player_b': env.EAST})
    assert env.state == initial_state, "State should not change when players attempt to move out of bounds"

    # Case 4: B has possession, A moves LEFT, B moves RIGHT
    initial_state = (0, 1, 3, 5, 1)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.WEST, 'player_b': env.EAST})
    assert env.state == initial_state, "State should not change when players attempt to move out of bounds"

    # Case 5: A has possession, A moves UP, B moves DOWN
    initial_state = (0, 1, 3, 5, 0)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.NORTH, 'player_b': env.SOUTH})
    assert env.state == initial_state, "State should not change when Player A attempts to move out of bounds"

    # Case 6: B has possession, A moves UP, B moves DOWN
    initial_state = (0, 1, 3, 5, 1)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.NORTH, 'player_b': env.SOUTH})
    assert env.state == initial_state, "State should not change when Player A attempts to move out of bounds"

    # Case 7: A has possession, A moves LEFT, B moves DOWN
    initial_state = (0, 1, 3, 5, 0)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.WEST, 'player_b': env.SOUTH})
    assert env.state == initial_state, "State should not change when Player A attempts to move out of bounds"

    # Case 8: B has possession, A moves LEFT, B moves DOWN
    initial_state = (0, 1, 3, 5, 1)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.WEST, 'player_b': env.SOUTH})
    assert env.state == initial_state, "State should not change when Player A attempts to move out of bounds"

    # Swap positions: A at right edge, B at top edge
    # Case 9: A has possession, A moves RIGHT, B moves UP
    initial_state = (3, 5, 0, 1, 0)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.NORTH})
    assert env.state == initial_state, "State should not change when players attempt to move out of bounds"

    # Case 10: B has possession, A moves RIGHT, B moves UP
    initial_state = (3, 5, 0, 1, 1)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.NORTH})
    assert env.state == initial_state, "State should not change when players attempt to move out of bounds"

    # Case 11: A has possession, A moves RIGHT, B moves LEFT
    initial_state = (3, 5, 0, 1, 0)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.WEST})
    assert env.state == initial_state, "State should not change when players attempt to move out of bounds"

    # Case 12: B has possession, A moves RIGHT, B moves LEFT
    initial_state = (3, 5, 0, 1, 1)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.WEST})
    assert env.state == initial_state, "State should not change when players attempt to move out of bounds"

    # Case 13: A has possession, A moves DOWN, B moves UP
    initial_state = (3, 5, 0, 1, 0)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.SOUTH, 'player_b': env.NORTH})
    assert env.state == initial_state, "State should not change when Player B attempts to move out of bounds"

    # Case 14: B has possession, A moves DOWN, B moves UP
    initial_state = (3, 5, 0, 1, 1)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.SOUTH, 'player_b': env.NORTH})
    assert env.state == initial_state, "State should not change when Player B attempts to move out of bounds"

    # Case 15: A has possession, A moves DOWN, B moves LEFT
    initial_state = (3, 5, 0, 1, 0)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.SOUTH, 'player_b': env.WEST})
    assert env.state == initial_state, "State should not change when Player B attempts to move out of bounds"

    # Case 16: B has possession, A moves DOWN, B moves LEFT
    initial_state = (3, 5, 0, 1, 1)
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.SOUTH, 'player_b': env.WEST})
    assert env.state == initial_state, "State should not change when Player B attempts to move out of bounds"

    # GOAL BOUNDARIES WITHOUT POSSESSION
    # Test Player A at left goal boundary without possession (row 1)
    env.reset()
    initial_state = (1, 1, 3, 3, 1)  # B has possession
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.WEST, 'player_b': env.NOOP})
    assert env.state == initial_state, "Player A should not enter left goal area without possession (row 1)"

    # Test Player A at left goal boundary without possession (row 2)
    env.reset()
    initial_state = (2, 1, 3, 3, 1)  # B has possession
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.WEST, 'player_b': env.NOOP})
    assert env.state == initial_state, "Player A should not enter left goal area without possession (row 2)"

    # Test Player B at right goal boundary without possession (row 1)
    env.reset()
    initial_state = (3, 3, 1, 5, 0)  # A has possession
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.NOOP, 'player_b': env.EAST})
    assert env.state == initial_state, "Player B should not enter right goal area without possession (row 1)"

    # Test Player B at right goal boundary without possession (row 2)
    env.reset()
    initial_state = (3, 3, 2, 5, 0)  # A has possession
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.NOOP, 'player_b': env.EAST})
    assert env.state == initial_state, "Player B should not enter right goal area without possession (row 2)"

    # Test Player B at left goal boundary without possession (row 1)
    env.reset()
    initial_state = (3, 3, 1, 1, 0)  # A has possession
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.NOOP, 'player_b': env.WEST})
    assert env.state == initial_state, "Player B should not move beyond left goal boundary without possession (row 1)"

    # Test Player B at left goal boundary without possession (row 2)
    env.reset()
    initial_state = (3, 3, 2, 1, 0)  # A has possession
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.NOOP, 'player_b': env.WEST})
    assert env.state == initial_state, "Player B should not move beyond left goal boundary without possession (row 2)"

    # Test Player A at right goal boundary without possession (row 1)
    env.reset()
    initial_state = (1, 5, 3, 3, 1)  # B has possession
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.NOOP})
    assert env.state == initial_state, "Player A should not move beyond right goal boundary without possession (row 1)"

    # Test Player A at right goal boundary without possession (row 2)
    env.reset()
    initial_state = (2, 5, 3, 3, 1)  # B has possession
    env.state = initial_state
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.NOOP})
    assert env.state == initial_state, "Player A should not move beyond right goal boundary without possession (row 2)"

def test_render(env, capsys):
    env.reset()
    env.render()
    captured = capsys.readouterr()
    assert "Player A position" in captured.out
    assert "Player B position" in captured.out
    assert "Ball possession" in captured.out

def test_possession_change_non_collision(env):
    # Test that possession doesn't change when players move without colliding
    env.state = (1, 1, 3, 3, 0)  # Player A has possession
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.WEST})
    assert env.state[4] == 0, "Possession should not change without collision"

    env.state = (1, 1, 3, 3, 1)  # Player B has possession
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.WEST})
    assert env.state[4] == 1, "Possession should not change without collision"

def test_simultaneous_goal_attempts(env):
    # Both players attempt to score simultaneously
    env.state = (1, 5, 1, 1, 0)  # A with ball near B's goal, B near A's goal
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.WEST})
    assert done['player_a'] and done['player_b'], "Game should end"
    assert reward['player_a'] == 1 and reward['player_b'] == -1, "Only A should score"

    env.reset()
    env.state = (1, 5, 1, 1, 1)  # B with ball near A's goal, A near B's goal
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.WEST})
    assert done['player_a'] and done['player_b'], "Game should end"
    assert reward['player_a'] == -1 and reward['player_b'] == 1, "Only B should score"

def test_edge_case_possession(env):
    # Test possession change when moving to the same cell from different distances
    env.state = (1, 1, 1, 2, 0)  # A has ball, both move right
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.EAST})
    assert env.state[4] == 0, "A should keep possession as it's closer"

    env.state = (1, 1, 1, 2, 1)  # B has ball, both move right
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.EAST})
    assert env.state[4] == 1, "B should keep possession even though A moves to the same cell"

    # Test possession change when moving to the same cell from different distances
    env.state = (1, 1, 1, 3, 0)  # A has ball, both move right
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.EAST})
    assert env.state[4] == 0, "A should keep possession as it's closer"

    env.state = (1, 1, 1, 3, 1)  # B has ball, both move right
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.EAST})
    assert env.state[4] == 1, "B should keep possession even though A moves to the same cell"

def test_multiple_consecutive_collisions(env):
    initial_state = (1, 2, 1, 3, 0)  # A has ball, players adjacent
    n_samples = 1000
    collision_count = 0
    possession_changes = 0
    last_possession = 0

    for _ in range(n_samples):
        env.state = initial_state
        obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.WEST})
        
        if env.state[1] == initial_state[1] and env.state[3] == initial_state[3]:
            collision_count += 1
        
        if env.state[4] != last_possession:
            possession_changes += 1
        
        last_possession = env.state[4]

    assert collision_count == n_samples, f"All steps should result in collision, got {collision_count}"
    possession_ratio = possession_changes / n_samples
    assert 0.45 <= possession_ratio <= 0.55, f"Possession should change roughly half the time, got {possession_ratio:.2f}"

def test_simultaneous_out_of_bounds(env):
    # Both players try to move out of bounds simultaneously
    env.state = (0, 1, 3, 5, 0)  # A at top edge, B at right edge
    initial_state = env.state
    obs, reward, done, truncated, info = env.step({'player_a': env.NORTH, 'player_b': env.EAST})
    assert env.state == initial_state, "State should not change when both players attempt to move out of bounds"

    # One player tries to move out of bounds, the other moves validly
    env.state = (0, 1, 3, 4, 1)  # A at top edge, B has possession
    obs, reward, done, truncated, info = env.step({'player_a': env.NORTH, 'player_b': env.EAST})
    assert env.state[3] == 5, "B should move right"
    assert env.state[0] == 0 and env.state[1] == 1, "A should not move"

def test_edge_case_goal_scoring(env):
    # Test scoring from the edge of the goal area
    env.state = (1, 5, 3, 3, 0)  # A with ball, at edge of B's goal
    obs, reward, done, truncated, info = env.step({'player_a': env.EAST, 'player_b': env.NOOP})
    assert done['player_a'] and done['player_b'], "Game should end"
    assert reward['player_a'] == 1 and reward['player_b'] == -1, "A should score"

    # Test scoring from the edge of own goal area
    env.reset()
    env.state = (2, 1, 3, 3, 0)  # A with ball, at edge of own goal
    obs, reward, done, truncated, info = env.step({'player_a': env.WEST, 'player_b': env.NOOP})
    assert done['player_a'] and done['player_b'], "Game should end"
    assert reward['player_a'] == -1 and reward['player_b'] == 1, "A should score an own goal"