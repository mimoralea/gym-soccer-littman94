import numpy as np
import random
from gym.envs.toy_text.utils import categorical_sample

class SoccerGridWorld:
    # Define constants for actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAND = 4

    def __init__(self, width=5, height=4, slip_prob=0.2, isd_possession_a=0.5, simultaneous_action=True, player_a_policy=None, player_b_policy=None):
        assert width >= 5, "Width must be at least 5 columns."
        assert height >= 4, "Height must be at least 4 rows."

        self.width = width + 2  # +2 for the columns where goals are located
        self.height = height
        self.slip_prob = slip_prob
        self.isd_possession_a = isd_possession_a
        self.simultaneous_action = simultaneous_action
        self.player_a_policy = player_a_policy
        self.player_b_policy = player_b_policy
        self.np_random = np.random.RandomState()
        
        # Initialize the state space and action space
        self.n_states = self.width * self.height * 2  # width * height * 2 (possession)
        self.n_actions = 5  # Actions: UP, DOWN, LEFT, RIGHT, STAND
        
        # Define the initial state distribution
        self.isd = self.generate_isd()
        
        # Define transition dynamics
        self.P = self._initialize_transition_dynamics()
        
        # Initialize current state
        self.s = self.reset()[0]
        self.lastaction = None
        
        # For alternating case
        if not self.simultaneous_action:
            self.current_player = None  # Will be set in reset()

    def generate_isd(self):
        distribution = []
        col_a = 2  # Player A starts 2 columns from their goal
        col_b = self.width - 3  # Player B starts 2 columns from their goal

        if self.height % 2 == 1:
            # Odd height: both players start in the middle row
            middle_row = self.height // 2
            for possession in range(2):  # 0: A, 1: B
                if self.simultaneous_action:
                    state = (middle_row, col_a, middle_row, col_b, possession)
                else:
                    for who_moves_first in range(2):  # 0: A, 1: B
                        state = (middle_row, col_a, middle_row, col_b, possession, who_moves_first)
                        distribution.append((0.25, state))
                if self.simultaneous_action:
                    distribution.append((0.5, state))
        else:
            # Even height: players start in different rows around the middle
            row_a_options = [self.height // 2 - 1, self.height // 2]
            row_b_options = [self.height // 2, self.height // 2 - 1]
            for i in range(2):
                row_a = row_a_options[i]
                row_b = row_b_options[i]
                for possession in range(2):  # 0: A, 1: B
                    if self.simultaneous_action:
                        state = (row_a, col_a, row_b, col_b, possession)
                        distribution.append((0.25, state))
                    else:
                        for who_moves_first in range(2):  # 0: A, 1: B
                            state = (row_a, col_a, row_b, col_b, possession, who_moves_first)
                            distribution.append((0.125, state))
        
        return distribution

    def _initialize_transition_dynamics(self):
        P = {}
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # UP, DOWN, LEFT, RIGHT, STAND
        
        for row_a in range(self.height):
            for col_a in range(self.width):
                for row_b in range(self.height):
                    for col_b in range(self.width):
                        for possession in range(2):  # 0: A, 1: B
                            if self.simultaneous_action:
                                state = (row_a, col_a, row_b, col_b, possession)
                                P[state] = {}
                                # Simultaneous action dynamics
                                for action_a in range(self.n_actions):
                                    for action_b in range(self.n_actions):
                                        transitions = []
                                        next_state, reward, done = self._get_next_state(state, action_a, action_b)
                                        transitions.append((1 - self.slip_prob, next_state, reward, done))
                                        
                                        # Handle slips in orthogonal directions
                                        orthogonal_moves_a = [(-self.directions[action_a][1], self.directions[action_a][0]), (self.directions[action_a][1], -self.directions[action_a][0])]
                                        orthogonal_moves_b = [(-self.directions[action_b][1], self.directions[action_b][0]), (self.directions[action_b][1], -self.directions[action_b][0])]
                                        for orth_move_a in orthogonal_moves_a:
                                            for orth_move_b in orthogonal_moves_b:
                                                slip_state, _, _ = self._get_next_state(state, 
                                                    self._action_from_direction(orth_move_a), 
                                                    self._action_from_direction(orth_move_b))
                                                transitions.append((self.slip_prob / 4, slip_state, reward, done))
                                        P[state][(action_a, action_b)] = transitions
                            else:
                                # Alternating action dynamics
                                for who_moves_next in [0, 1]:  # 0: Player A, 1: Player B
                                    state = (row_a, col_a, row_b, col_b, possession, who_moves_next)
                                    P[state] = {}
                                    for action in range(self.n_actions):
                                        transitions = []
                                        next_state, reward, done = self._get_next_state(state, action, None)
                                        next_state = (*next_state[:5], 1 - who_moves_next)  # Switch to other player's turn
                                        transitions.append((1 - self.slip_prob, next_state, reward, done))
                                        
                                        # Handle slips in orthogonal directions
                                        orthogonal_moves = [(-self.directions[action][1], self.directions[action][0]), (self.directions[action][1], -self.directions[action][0])]
                                        for orth_move in orthogonal_moves:
                                            slip_action = self._action_from_direction(orth_move)
                                            slip_state, slip_reward, slip_done = self._get_next_state(state, slip_action, None)
                                            slip_state = (*slip_state[:5], 1 - who_moves_next)  # Switch to other player's turn
                                            transitions.append((self.slip_prob / 2, slip_state, slip_reward, slip_done))
                                        P[state][action] = transitions
        
        return P

    def _get_next_state(self, state, action_a, action_b):
        if self.simultaneous_action:
            row_a, col_a, row_b, col_b, possession = state
        else:
            row_a, col_a, row_b, col_b, possession, who_moves_next = state
        
        # Handle actions and slip probability
        def move(row, col, action):
            if action is None:
                return row, col
            intended_move = self.directions[action]
            new_row = max(0, min(self.height - 1, row + intended_move[0]))
            new_col = max(0, min(self.width - 1, col + intended_move[1]))
            return new_row, new_col
        
        # Update positions based on actions
        if self.simultaneous_action or who_moves_next == 0:
            next_row_a, next_col_a = move(row_a, col_a, action_a)
        else:
            next_row_a, next_col_a = row_a, col_a
        
        if self.simultaneous_action or who_moves_next == 1:
            next_row_b, next_col_b = move(row_b, col_b, action_b)
        else:
            next_row_b, next_col_b = row_b, col_b
        
        # Handle STAND action properly in alternating action case
        if not self.simultaneous_action:
            if who_moves_next == 0:  # Player A's turn
                if action_a == self.STAND:
                    next_row_a, next_col_a = row_a, col_a
                elif (next_row_a, next_col_a) == (row_b, col_b):
                    possession = 1  # Player B gains possession
                    next_row_a, next_col_a = row_a, col_a
            else:  # Player B's turn
                if action_b == self.STAND:
                    next_row_b, next_col_b = row_b, col_b
                elif (next_row_b, next_col_b) == (row_a, col_a):
                    possession = 0  # Player A gains possession
                    next_row_b, next_col_b = row_b, col_b
        
        # Check for goals and terminal state
        done = False
        reward = 0
        if possession == 0:  # Player A has the ball
            if next_col_a == 0:  # Player A scores in its own goal (own goal)
                done = True
                reward = -1  # Negative reward for own goal
            elif next_col_a == self.width - 1:  # Player A scores in opponent's goal
                done = True
                reward = 1
            elif next_row_a == row_b and next_col_a == col_b:  # Player B steals
                possession = 1
        else:  # Player B has the ball
            if next_col_b == self.width - 1:  # Player B scores in its own goal (own goal)
                done = True
                reward = 1  # Positive reward for player A when B scores own goal
            elif next_col_b == 0:  # Player B scores in opponent's goal
                done = True
                reward = -1
            elif next_row_b == row_a and next_col_b == col_a:  # Player A steals
                possession = 0
        
        # Handle simultaneous action collision
        if self.simultaneous_action and (next_row_a == next_row_b and next_col_a == next_col_b):
            if action_a == self.STAND and action_b != self.STAND:
                possession = 0  # Player A gains possession
            elif action_b == self.STAND and action_a != self.STAND:
                possession = 1  # Player B gains possession
            elif action_a != self.STAND and action_b != self.STAND:
                possession = self.np_random.choice([0, 1])  # Randomly decide who gets possession
        
        next_state = (next_row_a, next_col_a, next_row_b, next_col_b, possession)
        
        return next_state, reward, done

    def step(self, action):
        if self.simultaneous_action:
            action_a, action_b = action
            transitions = self.P[self.s][(action_a, action_b)]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            prob, next_state, reward, done = transitions[i]
            self.s = next_state
            self.lastaction = action
            obs = {
                "player_a": (next_state[0], next_state[1], next_state[2], next_state[3], 1 if next_state[4] == 0 else 0),
                "player_b": (next_state[2], next_state[3], next_state[0], next_state[1], 1 if next_state[4] == 1 else 0)
            }
            rewards = {"player_a": reward, "player_b": -reward}
            dones = {"player_a": done, "player_b": done}
            truncateds = {"player_a": False, "player_b": False}
            infos = {"player_a": {"prob": prob}, "player_b": {"prob": prob}}
            return obs, rewards, dones, truncateds, infos
        else:
            transitions = self.P[self.s][action]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            prob, next_state, reward, done = transitions[i]
            self.s = next_state
            self.lastaction = action
            self.current_player = next_state[5]  # Update current player
            
            current_player_name = "player_a" if self.current_player == 0 else "player_b"
            obs = (next_state[0], next_state[1], next_state[2], next_state[3], 1 if next_state[4] == self.current_player else 0)
            reward = reward if self.current_player == 0 else -reward
            
            return obs, reward, done, False, {"prob": prob}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random.seed(seed)
        i = categorical_sample([is_[0] for is_ in self.isd], self.np_random)
        p, self.s = self.isd[i]
        self.lastaction = None
        if self.simultaneous_action:
            obs = {
                "player_a": (self.s[0], self.s[1], self.s[2], self.s[3], 1 if self.s[4] == 0 else 0),
                "player_b": (self.s[2], self.s[3], self.s[0], self.s[1], 1 if self.s[4] == 1 else 0)
            }
        else:
            self.current_player = self.s[5]
            obs = (self.s[0], self.s[1], self.s[2], self.s[3], 1 if self.s[4] == self.current_player else 0)
        return obs, {"prob": p}

    def _action_from_direction(self, direction):
        return self.directions.index(direction)
