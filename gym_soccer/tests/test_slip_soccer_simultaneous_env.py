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
    def run_scoring_test(initial_state, action_a, action_b, iterations=100000):
        score_count = 0
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            action = {'player_a': action_a, 'player_b': action_b}
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated['player_a'] or terminated['player_b']:
                assert abs(reward['player_a']) == 1 and abs(reward['player_b']) == 1, "Both players must receive a reward/penalty for a goal"
                score_count += 1

        score_ratio = score_count / iterations
        print(f"Score ratio: {score_ratio:.2f}")
        assert 0.75 <= score_ratio <= 0.85, f"Score ratio: {score_ratio:.2f}, expected close to 0.8"

    # Test Player A scoring
    run_scoring_test((1, 5, 3, 1, 0), env.EAST, env.NOOP)

    # Test Player B scoring
    run_scoring_test((3, 5, 1, 1, 1), env.NOOP, env.WEST)

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
    action = {'player_a': env.EAST, 'player_b': env.WEST}
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.state[4] == 0, "Possession should not change without collision"

    env.reset()
    env.state = (1, 1, 3, 3, 1)  # Player B has possession
    action = {'player_a': env.EAST, 'player_b': env.WEST}
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.state[4] == 1, "Possession should not change without collision"

def test_slip_into_goal(env):
    def run_slip_goal_test(initial_state, action_a, action_b, iterations=100000):
        goal_count = 0
        for _ in range(iterations):
            env.reset()
            env.state = initial_state
            action = {'player_a': action_a, 'player_b': action_b}
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated['player_a'] or terminated['player_b']:
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
            obs, reward, done, truncated, info = env.step({'player_a': action_a, 'player_b': action_b})
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
            obs, reward, done, truncated, info = env.step({'player_a': action, 'player_b': env.NOOP})
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
            obs, reward, done, truncated, info = env.step({'player_a': action_a, 'player_b': action_b})
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
        obs, reward, done, truncated, info = env.step({'player_a': env.NOOP, 'player_b': env.NOOP})
        if env.state != initial_state:
            slip_count += 1

    assert slip_count == 0, f"Expected no slips on STAND action, got {slip_count} slips"
