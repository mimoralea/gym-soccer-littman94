import pytest
import numpy as np
from gym_soccer.envs import SoccerSimultaneousEnv

@pytest.fixture
def env():
    return SoccerSimultaneousEnv(width=5, height=4, slip_prob=0.2)

@pytest.fixture(autouse=True)
def reset_env(env):
    env.reset()
    yield

def test_initialization(env):
    env.reset()
    assert env.width == 7  # 5 + 2 for goal columns
    assert env.height == 4
    assert env.slip_prob == 0.2
    assert env.action_space[0].n == 5
    # assert env.observation_space.n == 7 * 4 * 7 * 4 * 2

def test_reset(env):
    obs, info = env.reset()
    assert isinstance(obs, tuple)
    assert len(obs) == 2
    assert isinstance(info, tuple)
    assert 'p' in info[0]

def test_step(env):
    env.reset()
    action = (env.NOOP, env.NOOP)
    obs, reward, done, truncated, info = env.step(action)
    assert isinstance(obs, tuple)
    assert isinstance(reward, tuple)
    assert isinstance(done, tuple)
    assert isinstance(truncated, tuple)
    assert isinstance(info, tuple)

def test_scoring(env):
    def run_scoring_test(initial_state, action_a, action_b, iterations=100000):
        score_count = 0
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))
            if done[0] or done[1]:
                assert abs(reward[0]) == 1 and abs(reward[1]) == 1, "Both players must receive a reward/penalty for a goal"
                score_count += 1

        score_ratio = score_count / iterations
        print(f"Score ratio: {score_ratio:.2f}")
        assert 0.75 <= score_ratio <= 0.85, f"Score ratio: {score_ratio:.2f}, expected close to 0.8"

    # Test Player A scoring
    run_scoring_test((1, 5, 3, 1, 0), env.EAST, env.NOOP)

    # Test Player B scoring
    run_scoring_test((3, 5, 1, 1, 1), env.NOOP, env.WEST)

def test_own_goals(env):
    def run_own_goal_test(initial_state, action_a, action_b, iterations=100000):
        own_goal_count = 0
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))
            if done[0] and done[1]:
                if initial_state[4] == 0:  # Player A has the ball
                    if reward[0] == -1 and reward[1] == 1:
                        own_goal_count += 1
                else:  # Player B has the ball
                    if reward[0] == 1 and reward[1] == -1:
                        own_goal_count += 1

        own_goal_ratio = own_goal_count / iterations
        expected_ratio = 1 - env.slip_prob  # Expected ratio is 1 minus the slip probability
        assert np.isclose(own_goal_ratio, expected_ratio, atol=0.02), f"Own goal ratio: {own_goal_ratio:.2f}, expected close to {expected_ratio:.2f}"

    # Test Player A scoring an own goal (row 1)
    run_own_goal_test((1, 1, 3, 5, 0), env.WEST, env.NOOP)

    # Test Player A scoring an own goal (row 2)
    run_own_goal_test((2, 1, 3, 5, 0), env.WEST, env.NOOP)

    # Test Player B scoring an own goal (row 1)
    run_own_goal_test((3, 1, 1, 5, 1), env.NOOP, env.EAST)

    # Test Player B scoring an own goal (row 2)
    run_own_goal_test((3, 1, 2, 5, 1), env.NOOP, env.EAST)

def test_slip_probability(env):
    iterations = 100000
    intended_moves = 0
    orthogonal_moves = 0

    for _ in range(iterations):
        env.reset()
        env.state = (2, 2, 3, 3, 0)  # Player A in middle
        obs, reward, done, truncated, info = env.step((env.EAST, env.NOOP))

        if env.state[1] == 3:
            intended_moves += 1
        elif env.state[0] in [1, 3]:
            orthogonal_moves += 1

    intended_ratio = intended_moves / iterations
    orthogonal_ratio = orthogonal_moves / iterations

    assert 0.75 <= intended_ratio <= 0.85, f"Intended move ratio: {intended_ratio:.2f}, expected close to 0.8"
    assert 0.15 <= orthogonal_ratio <= 0.25, f"Orthogonal move ratio: {orthogonal_ratio:.2f}, expected close to 0.2"

def test_both_players_moving_collision(env):
    def run_move_collision_test(initial_state, action_a, action_b, iterations=100000):
        collision_count = 0
        possession_switch_count = 0
        initial_possession = initial_state[4]

        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))

            if (env.state[1] == initial_state[1] and env.state[3] == initial_state[3] and
                env.state[0] == initial_state[0] and env.state[2] == initial_state[2]):
                collision_count += 1

            if env.state[4] != initial_possession:
                possession_switch_count += 1

        collision_ratio = collision_count / iterations
        possession_switch_ratio = possession_switch_count / iterations

        # Expected collision_ratio: 80% chance of intended move for each player
        expected_collision = 0.8 * 0.8
        assert np.isclose(collision_ratio, expected_collision, atol=0.02), f"Collision ratio: {collision_ratio:.2f}, expected close to {expected_collision:.2f}"

        # Possession should switch 50% of the time when there's a collision
        expected_possession_switch = expected_collision * 0.5
        assert np.isclose(possession_switch_ratio, expected_possession_switch, atol=0.02), f"Possession switch ratio: {possession_switch_ratio:.2f}, expected close to {expected_possession_switch:.2f}"

    # Test horizontal collision (A on left, B on right)
    run_move_collision_test((1, 2, 1, 3, 0), env.EAST, env.WEST)
    run_move_collision_test((1, 2, 1, 3, 1), env.EAST, env.WEST)

    # Test vertical collision (A on top, B on bottom)
    run_move_collision_test((1, 3, 2, 3, 0), env.SOUTH, env.NORTH)
    run_move_collision_test((1, 3, 2, 3, 1), env.SOUTH, env.NORTH)

def test_one_player_standing_collision(env):
    def run_stand_collision_test(initial_state, action_a, action_b, iterations=100000):
        assert action_a == env.NOOP or action_b == env.NOOP, "One player must be standing"
        no_move_count = 0
        possession_switch_count = 0
        initial_possession = initial_state[4]

        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))

            if (env.state[1] == initial_state[1] and env.state[3] == initial_state[3] and
                env.state[0] == initial_state[0] and env.state[2] == initial_state[2]):
                no_move_count += 1

            if env.state[4] != initial_possession:
                possession_switch_count += 1

        no_move_ratio = no_move_count / iterations
        print(f"No move ratio: {no_move_ratio:.2f}")
        possession_switch_ratio = possession_switch_count / iterations
        print(f"Possession switch ratio: {possession_switch_ratio:.2f}")

        # Expected no_move_ratio: 80% chance of intended move * 100% chance of other player staying
        expected_no_move = 0.8 * 1.0
        assert np.isclose(no_move_ratio, expected_no_move, atol=0.02), f"No move ratio: {no_move_ratio:.2f}, expected close to {expected_no_move:.2f}"

        # Possession should switch whenever there's a collision (80% of the time)
        assert np.isclose(possession_switch_ratio, 0.8, atol=0.02), f"Possession switch ratio: {possession_switch_ratio:.2f}, expected close to 0.8"

    # Horizontal collisions
    # A on left, B on right
    run_stand_collision_test((1, 2, 1, 3, 0), env.EAST, env.NOOP)  # A moves, B stands
    run_stand_collision_test((1, 2, 1, 3, 1), env.EAST, env.NOOP)
    run_stand_collision_test((1, 2, 1, 3, 0), env.NOOP, env.WEST)   # A stands, B moves
    run_stand_collision_test((1, 2, 1, 3, 1), env.NOOP, env.WEST)

    # A on right, B on left
    run_stand_collision_test((1, 4, 1, 3, 0), env.WEST, env.NOOP)   # A moves, B stands
    run_stand_collision_test((1, 4, 1, 3, 1), env.WEST, env.NOOP)
    run_stand_collision_test((1, 4, 1, 3, 0), env.NOOP, env.EAST)  # A stands, B moves
    run_stand_collision_test((1, 4, 1, 3, 1), env.NOOP, env.EAST)

    # Vertical collisions
    # A on top, B on bottom
    run_stand_collision_test((1, 3, 2, 3, 0), env.SOUTH, env.NOOP)   # A moves, B stands
    run_stand_collision_test((1, 3, 2, 3, 1), env.SOUTH, env.NOOP)
    run_stand_collision_test((1, 3, 2, 3, 0), env.NOOP, env.NORTH)     # A stands, B moves
    run_stand_collision_test((1, 3, 2, 3, 1), env.NOOP, env.NORTH)

    # A on bottom, B on top
    run_stand_collision_test((2, 3, 1, 3, 0), env.NORTH, env.NOOP)     # A moves, B stands
    run_stand_collision_test((2, 3, 1, 3, 1), env.NORTH, env.NOOP)
    run_stand_collision_test((2, 3, 1, 3, 0), env.NOOP, env.SOUTH)   # A stands, B moves
    run_stand_collision_test((2, 3, 1, 3, 1), env.NOOP, env.SOUTH)

def test_move_to_same_cell_collision(env):
    def run_move_to_same_cell_collision_test(initial_state, action_a, action_b, iterations=100000):
        move_success_counts = {'A': 0, 'B': 0}
        possession_counts = {'A': 0, 'B': 0}
        possession_switch_count = 0
        initial_possession = initial_state[4]

        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))
            move_a = env.ACTION_INT_TO_MOVE[action_a]
            move_b = env.ACTION_INT_TO_MOVE[action_b]

            intended_row_a, intended_col_a = env._next_cell(initial_state[0], initial_state[1], move_a, initial_possession == 0)
            intended_row_b, intended_col_b = env._next_cell(initial_state[2], initial_state[3], move_b, initial_possession == 1)

            if env.state[0] == intended_row_a and env.state[1] == intended_col_a:
                move_success_counts['A'] += 1
            elif env.state[2] == intended_row_b and env.state[3] == intended_col_b:
                move_success_counts['B'] += 1

            if env.state[4] == 0:
                possession_counts['A'] += 1
            else:
                possession_counts['B'] += 1

            if env.state[4] != initial_possession:
                possession_switch_count += 1

        # Both players move to their intended cells 80% * 80%, then half we bounce A, half we bounce B 0.8 * 0.8 * 0.5 = 0.32
        # Plus the 80% we move to the intended cell, the other player slips 0.8 * 0.2 = 0.16
        # Total 0.32 + 0.16 = 0.48
        for player, count in move_success_counts.items():
            success_ratio = count / iterations
            print(f"Move success ratio for Player {player}: {success_ratio:.2f}")
            assert 0.47 <= success_ratio <= 0.49, f"Move success ratio for Player {player}: {success_ratio:.2f}, expected close to 0.48"

        # Possession should switch 50% of the time when there's a collision, with 0.8 * 0.8 * 0.5 = 0.32
        # Then there is a potential additional collision in cell 2,1, through slipping, with possession uniformly random
        # 0.1 * 0.1 * 0.5 = 0.005
        # Total 0.32 + 0.005 = 0.325
        possession_switch_ratio = possession_switch_count / iterations
        print(f"Possession switch ratio: {possession_switch_ratio:.2f}")
        assert 0.31 <= possession_switch_ratio <= 0.33, f"Possession switch ratio: {possession_switch_ratio:.2f}, expected close to 0.32"

        # Possession switches ~0.325, player retains possession 1-0.325 = ~0.675
        for player, count in possession_counts.items():
            possession_ratio = count / iterations
            print(f"Possession ratio for Player {player}: {possession_ratio:.2f}")
            if initial_possession == 0 and player == 'A':
                assert 0.67 <= possession_ratio <= 0.69, f"Possession ratio for Player {player}: {possession_ratio:.2f}, expected close to 0.68"
            elif initial_possession == 0 and player == 'B':
                assert 0.31 <= possession_ratio <= 0.33, f"Possession ratio for Player {player}: {possession_ratio:.2f}, expected close to 0.32"
            elif initial_possession == 1 and player == 'B':
                assert 0.67 <= possession_ratio <= 0.69, f"Possession ratio for Player {player}: {possession_ratio:.2f}, expected close to 0.68"
            elif initial_possession == 1 and player == 'A':
                assert 0.31 <= possession_ratio <= 0.33, f"Possession ratio for Player {player}: {possession_ratio:.2f}, expected close to 0.32"

    # Diagonal movements
    run_move_to_same_cell_collision_test((1, 1, 2, 2, 0), env.EAST, env.NORTH)
    run_move_to_same_cell_collision_test((1, 1, 2, 2, 1), env.EAST, env.NORTH)

    # Horizontal movements
    run_move_to_same_cell_collision_test((1, 1, 1, 3, 0), env.EAST, env.WEST)
    run_move_to_same_cell_collision_test((1, 1, 1, 3, 1), env.EAST, env.WEST)

def test_corner_bounce(env):
    def run_corner_test(initial_state, action_a, action_b, iterations=100000):
        player_move_count = {'A': 0, 'B': 0}
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))
            if env.state[0] != initial_state[0] or env.state[1] != initial_state[1]:
                player_move_count['A'] += 1
            if env.state[2] != initial_state[2] or env.state[3] != initial_state[3]:
                player_move_count['B'] += 1

        for player, count in player_move_count.items():
            slip_ratio = count / iterations
            print(f"Slip ratio for Player {player}: {slip_ratio:.4f}")
            # move is 10% chance of slipping for each player, so 90% chance of not slipping
            assert 0.09 <= slip_ratio <= 0.11, f"Slip ratio for Player {player}: {slip_ratio:.4f}, expected close to 0.1"

    # Test Player A at top edge, B at right edge
    run_corner_test((0, 1, 3, 5, 0), env.NORTH, env.EAST)
    run_corner_test((0, 1, 3, 5, 1), env.NORTH, env.EAST)
    run_corner_test((0, 1, 3, 5, 0), env.NORTH, env.SOUTH)
    run_corner_test((0, 1, 3, 5, 1), env.NORTH, env.SOUTH)
    run_corner_test((0, 1, 3, 5, 0), env.WEST, env.EAST)
    run_corner_test((0, 1, 3, 5, 1), env.WEST, env.EAST)
    run_corner_test((0, 1, 3, 5, 0), env.WEST, env.SOUTH)
    run_corner_test((0, 1, 3, 5, 1), env.WEST, env.SOUTH)

    # Test Player B at top edge, A at right edge
    run_corner_test((3, 5, 0, 1, 0), env.EAST, env.NORTH)
    run_corner_test((3, 5, 0, 1, 1), env.EAST, env.NORTH)
    run_corner_test((3, 5, 0, 1, 0), env.SOUTH, env.NORTH)
    run_corner_test((3, 5, 0, 1, 1), env.SOUTH, env.NORTH)
    run_corner_test((3, 5, 0, 1, 0), env.EAST, env.WEST)
    run_corner_test((3, 5, 0, 1, 1), env.EAST, env.WEST)
    run_corner_test((3, 5, 0, 1, 0), env.SOUTH, env.WEST)
    run_corner_test((3, 5, 0, 1, 1), env.SOUTH, env.WEST)

def test_goal_boundaries_no_goal(env):

    def run_goal_boundary_test(initial_state, action_a, action_b, iterations=100000):
        terminal_count = 0
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))
            if any(done) or any(truncated):
                terminal_count += 1

        assert terminal_count == 0, "No goal should be scored"

    # Test goal boundaries without possession
    run_goal_boundary_test((1, 1, 3, 3, 1), env.WEST, env.NOOP)  # A at left goal boundary
    run_goal_boundary_test((2, 1, 3, 3, 1), env.WEST, env.NOOP)  # A at left goal boundary
    run_goal_boundary_test((1, 5, 3, 3, 1), env.EAST, env.NOOP)  # A at right goal boundary
    run_goal_boundary_test((2, 5, 3, 3, 1), env.EAST, env.NOOP)  # A at right goal boundary

    run_goal_boundary_test((3, 3, 1, 1, 0), env.NOOP, env.WEST)  # B at left goal boundary
    run_goal_boundary_test((3, 3, 2, 1, 0), env.NOOP, env.WEST)  # B at left goal boundary
    run_goal_boundary_test((3, 3, 1, 5, 0), env.NOOP, env.EAST)  # B at right goal boundary
    run_goal_boundary_test((3, 3, 2, 5, 0), env.NOOP, env.EAST)  # B at right goal boundary

def test_render(env, capsys):
    env.reset()
    env.render()
    captured = capsys.readouterr()
    assert "Player A position" in captured.out
    assert "Player B position" in captured.out
    assert "Ball possession" in captured.out

def test_possession_change_non_collision(env):
    # Test that possession doesn't change when players move without colliding
    env.reset()
    env.state = (1, 1, 3, 3, 0)  # Player A has possession
    obs, reward, done, truncated, info = env.step((env.EAST, env.WEST))
    assert env.state[4] == 0, "Possession should not change without collision"

    env.reset()
    env.state = (1, 1, 3, 3, 1)  # Player B has possession
    obs, reward, done, truncated, info = env.step((env.EAST, env.WEST))
    assert env.state[4] == 1, "Possession should not change without collision"

def test_simultaneous_goal_attempts(env):
    for possession in [0, 1]:
        iterations = 100000
        a_score_count = 0
        b_score_count = 0

        for _ in range(iterations):
            env.reset()
            env.state = (1, 5, 1, 1, possession)  # one with ball near opponents's goal
            obs, reward, done, truncated, info = env.step((env.EAST, env.WEST))
            if reward[0] == 1:
                a_score_count += 1
            elif reward[1] == 1:
                b_score_count += 1

        a_score_ratio = a_score_count / iterations
        print(f"A score ratio: {a_score_ratio:.2f}")
        b_score_ratio = b_score_count / iterations
        print(f"B score ratio: {b_score_ratio:.2f}")

        if possession == 0:
            assert 0.75 <= a_score_ratio <= 0.85, f"A score ratio: {a_score_ratio:.2f}, expected close to 0.8"
            assert b_score_ratio == 0, f"B score ratio: {b_score_ratio:.2f}, expected 0"
        else:
            assert b_score_ratio <= 0.85, f"B score ratio: {b_score_ratio:.2f}, expected close to 0.8"
            assert a_score_ratio == 0, f"A score ratio: {a_score_ratio:.2f}, expected 0"

def test_edge_case_possession(env):
    # Test possession change when moving to the same cell from different distances
    env.reset()
    env.state = (1, 1, 1, 2, 0)  # A has ball, both move right
    obs, reward, done, truncated, info = env.step((env.EAST, env.EAST))
    assert env.state[4] == 0, "A should keep possession as it's closer"

    env.reset()
    env.state = (1, 1, 1, 2, 1)  # B has ball, both move right
    obs, reward, done, truncated, info = env.step((env.EAST, env.EAST))
    assert env.state[4] == 1, "B should keep possession even though A moves to the same cell"

    # Test possession change when moving to the same cell from different distances
    env.reset()
    env.state = (1, 1, 1, 3, 0)  # A has ball, both move right
    obs, reward, done, truncated, info = env.step((env.EAST, env.EAST))
    assert env.state[4] == 0, "A should keep possession as it's closer"

    env.reset()
    env.state = (1, 1, 1, 3, 1)  # B has ball, both move right
    obs, reward, done, truncated, info = env.step((env.EAST, env.EAST))
    assert env.state[4] == 1, "B should keep possession even though A moves to the same cell"

def test_multiple_consecutive_collisions(env):
    initial_state = (1, 2, 1, 3, 0)  # A has ball, players adjacent
    n_samples = 100000
    collision_count = 0
    possession_after_collision_count = {
        'A': 0,
        'B': 0
    }
    for _ in range(n_samples):
        env.reset()
        env.state = initial_state
        obs, reward, done, truncated, info = env.step((env.EAST, env.WEST))

        if env.state[1] == initial_state[1] and env.state[3] == initial_state[3] and \
            env.state[0] == initial_state[0] and env.state[2] == initial_state[2]:
            collision_count += 1
            if env.state[4] == 0:
                possession_after_collision_count['A'] += 1
            else:
                possession_after_collision_count['B'] += 1

    collision_ratio = collision_count / n_samples
    expected_collision_ratio = 0.8 * 0.8
    assert np.isclose(collision_ratio, expected_collision_ratio, atol=0.02), f"Collision ratio: {collision_ratio:.2f}, expected close to {expected_collision_ratio:.2f}"
    for player, count in possession_after_collision_count.items():
        possession_ratio = count / collision_count
        expected_possession_ratio = 0.5
        assert np.isclose(possession_ratio, expected_possession_ratio, atol=0.02), f"Possession ratio for {player}: {possession_ratio:.2f}, expected close to {expected_possession_ratio:.2f}"

def test_slip_into_goal(env):
    def run_slip_goal_test(initial_state, action_a, action_b, iterations=100000):
        goal_count = 0
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))
            if done[0] or done[1]:
                goal_count += 1

        goal_ratio = goal_count / iterations
        assert 0.09 <= goal_ratio <= 0.11, f"Goal ratio: {goal_ratio:.2f}, expected close to 0.1"

    # Test A slipping into own goal
    run_slip_goal_test((1, 1, 3, 3, 0), env.NORTH, env.NOOP)
    run_slip_goal_test((2, 1, 3, 3, 0), env.NORTH, env.NOOP)
    run_slip_goal_test((1, 1, 3, 3, 0), env.SOUTH, env.NOOP)
    run_slip_goal_test((2, 1, 3, 3, 0), env.SOUTH, env.NOOP)

    # Test A slipping into B's goal
    run_slip_goal_test((1, 5, 3, 3, 0), env.NORTH, env.NOOP)
    run_slip_goal_test((2, 5, 3, 3, 0), env.NORTH, env.NOOP)
    run_slip_goal_test((1, 5, 3, 3, 0), env.SOUTH, env.NOOP)
    run_slip_goal_test((2, 5, 3, 3, 0), env.SOUTH, env.NOOP)

    # Test B slipping into A's goal
    run_slip_goal_test((3, 3, 1, 1, 1), env.NOOP, env.NORTH)
    run_slip_goal_test((3, 3, 2, 1, 1), env.NOOP, env.NORTH)
    run_slip_goal_test((3, 3, 1, 1, 1), env.NOOP, env.SOUTH)
    run_slip_goal_test((3, 3, 2, 1, 1), env.NOOP, env.SOUTH)

    # Test B slipping into own goal
    run_slip_goal_test((3, 3, 1, 5, 1), env.NOOP, env.NORTH)
    run_slip_goal_test((3, 3, 2, 5, 1), env.NOOP, env.NORTH)
    run_slip_goal_test((3, 3, 1, 5, 1), env.NOOP, env.SOUTH)
    run_slip_goal_test((3, 3, 2, 5, 1), env.NOOP, env.SOUTH)

def test_bounce_off_horizontal_edges(env):
    def run_bounce_test(initial_state, action_a, action_b, iterations=100000):
        bounce_count = 0
        slip_count = 0
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))
            if env.state == initial_state:
                bounce_count += 1
            elif env.state != initial_state:
                slip_count += 1

        bounce_ratio = bounce_count / iterations
        slip_ratio = slip_count / iterations
        assert 0.79 <= bounce_ratio <= 0.81, f"Bounce ratio: {bounce_ratio:.2f}, expected close to 0.8"
        assert 0.19 <= slip_ratio <= 0.21, f"Slip ratio: {slip_ratio:.2f}, expected close to 0.2"

    # Test bouncing off top edge
    run_bounce_test((0, 2, 3, 3, 0), env.NORTH, env.NOOP)
    run_bounce_test((0, 3, 3, 3, 0), env.NORTH, env.NOOP)
    run_bounce_test((3, 3, 0, 2, 1), env.NOOP, env.NORTH)
    run_bounce_test((3, 3, 0, 3, 1), env.NOOP, env.NORTH)

    # Test bouncing off bottom edge
    run_bounce_test((3, 2, 0, 3, 0), env.SOUTH, env.NOOP)
    run_bounce_test((3, 3, 0, 3, 0), env.SOUTH, env.NOOP)
    run_bounce_test((0, 3, 3, 2, 0), env.NOOP, env.SOUTH)
    run_bounce_test((0, 3, 3, 3, 0), env.NOOP, env.SOUTH)

def test_bounce_off_corner_edges(env):
    def run_bounce_test(initial_state, action, iterations=100000):
        bounce_count = 0
        slip_count = 0
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action, env.NOOP))
            if env.state == initial_state:
                bounce_count += 1
            elif env.state != initial_state:
                slip_count += 1

        bounce_ratio = bounce_count / iterations
        slip_ratio = slip_count / iterations
        assert 0.89 <= bounce_ratio <= 0.91, f"Bounce ratio: {bounce_ratio:.2f}, expected close to 0.9"
        assert 0.09 <= slip_ratio <= 0.11, f"Slip ratio: {slip_ratio:.2f}, expected close to 0.1"

    # Test bouncing off left edge (non-goal row)
    run_bounce_test((0, 1, 3, 3, 1), env.WEST)

    # Test bouncing off right edge (non-goal row)
    run_bounce_test((3, 5, 0, 3, 1), env.EAST)

def test_collision_through_slip(env):
    def run_slip_collision_test(initial_state, action_a, action_b, iterations=100000):
        collision_count = 0
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            obs, reward, done, truncated, info = env.step((action_a, action_b))
            if env.state[0] == initial_state[0] and env.state[1] == initial_state[1] and \
                env.state[2] == initial_state[2] and env.state[3] == initial_state[3]:
                collision_count += 1

        collision_ratio = collision_count / iterations
        expected_ratio = 0.1  # 10% chance of slip for one player, other player moves as intended
        assert np.isclose(collision_ratio, expected_ratio, atol=0.02), f"Collision ratio: {collision_ratio:.2f}, expected close to {expected_ratio:.2f}"

    # Test A slipping into B's cell
    run_slip_collision_test((2, 2, 2, 3, 0), env.NORTH, env.NOOP)
    run_slip_collision_test((2, 2, 2, 3, 1), env.NORTH, env.NOOP)

    # Test B slipping into A's cell
    run_slip_collision_test((2, 3, 2, 2, 0), env.NOOP, env.NORTH)
    run_slip_collision_test((2, 3, 2, 2, 1), env.NOOP, env.NORTH)

def test_no_slip_on_stand(env):
    initial_state = (1, 2, 3, 4, 0)
    iterations = 100000
    slip_count = 0

    for _ in range(iterations):
        env.reset()
        env.state = initial_state
        obs, reward, done, truncated, info = env.step((env.NOOP, env.NOOP))
        if env.state != initial_state:
            slip_count += 1

    assert slip_count == 0, f"Expected no slips on STAND action, got {slip_count} slips"
