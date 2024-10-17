import numpy as np
from gym_soccer.envs.soccer_simultaneous_env import SoccerSimultaneousEnv

def get_random_policy(n_states=761, n_actions=5, seed=0):
    random_policy = {}
    random_state = np.random.RandomState(seed)
    for s in range(n_states):
        random_policy[s] = random_state.randint(0, n_actions)
    return random_policy

def get_stand_policy(n_states=761):
    stand_policy = {}
    for s in range(n_states):
        stand_policy[s] = SoccerSimultaneousEnv.NOOP
    return stand_policy

def save_policy(policy, filename, mode='wb'):
    import pickle
    assert isinstance(policy, dict), "Policy must be a dictionary"
    # Save dictionary to a file
    with open(filename, mode) as f:
        pickle.dump(policy, f)

def load_policy(filename, mode='rb'):
    import pickle
    with open(filename, mode) as f:
        return pickle.load(f)
