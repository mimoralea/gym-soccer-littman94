import numpy as np
from gym.envs.toy_text.utils import categorical_sample
from gym import spaces

class SoccerSimultaneousEnv:
    # Define constants for actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAND = 4
    ACTION_TO_MOVE = {
        UP: (0, -1),
        DOWN: (0, 1),
        LEFT: (-1, 0),
        RIGHT: (1, 0),
        STAND: (0, 0)
    }
    MOVE_TO_ACTION = {v: k for k, v in ACTION_TO_MOVE.items()}
    ACTION_STRING = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAND']


    def __init__(self, width=5, height=4, slip_prob=0.0, player_a_policy=None, player_b_policy=None):

        # Minimum pitch size is 5x4
        assert width >= 5, "Width must be at least 5 columns."
        assert height >= 4, "Height must be at least 4 rows."

        self.width = width + 2  # +2 for the columns where goals are located
        self.height = height
        self.goal_rows = ((height - 1) // 2, height // 2) if height % 2 == 0 else (height // 2 - 1, height // 2, height // 2 + 1)
        self.slip_prob = slip_prob
        self.player_a_policy = player_a_policy
        self.player_b_policy = player_b_policy
        self.np_random = np.random.RandomState()

        # Initialize the state space and action space
        self.n_states = self.width * self.height * self.width * self.height * 2  # width * height * width * height * 2 (possession)
        self.n_actions = len(self.ACTION_TO_MOVE)  # Actions: UP, DOWN, LEFT, RIGHT, STAND

        # Define the initial state distribution
        self.isd = self._generate_isd()

        # Define transition dynamics and create observation cache
        self.P = self._initialize_transition_dynamics()

        # Update observation space to be Discrete
        self.observation_space = spaces.Discrete(self.width * self.height * self.width * self.height * 2)
        self.action_space = spaces.Discrete(self.n_actions)

        # Add a flag to track if reset has been called
        self.reset_called = False
        # Add a flag to track if the episode is done or truncated
        self.episode_ended = False

        # Initialize self.state as a dictionary
        self.state = None

        # Replace self.s with self.observations
        self.observations = None

    def _generate_isd(self):
        distribution = []
        col_a = 2  # Player A starts 2 columns from their goal
        col_b = self.width - 3  # Player B starts 2 columns from their goal

        if len(self.goal_rows) % 2 == 0:  # Even number of goal rows
            middle_index = len(self.goal_rows) // 2
            row_options = [self.goal_rows[middle_index - 1], self.goal_rows[middle_index]]
            for row_a in row_options:
                row_b = row_options[1] if row_a == row_options[0] else row_options[0]
                for possession in range(2):  # 0: A, 1: B
                    state = (row_a, col_a, row_b, col_b, possession)
                    distribution.append((0.25, state))
        else:  # Odd number of goal rows
            middle_row = self.goal_rows[len(self.goal_rows) // 2]
            for possession in range(2):  # 0: A, 1: B
                state = (middle_row, col_a, middle_row, col_b, possession)
                distribution.append((0.5, state))

        return distribution

    def _initialize_transition_dynamics(self):
        P = {}

        for row_a in range(self.height):
            for col_a in range(self.width):
                for row_b in range(self.height):
                    for col_b in range(self.width):
                        for possession in range(2):  # 0: A, 1: B
                            state = self._game_state_to_dict((row_a, col_a, row_b, col_b, possession))
                            P[tuple(state.values())] = {}
                            # Simultaneous action dynamics
                            for action_a in range(self.n_actions):
                                for action_b in range(self.n_actions):
                                    transitions = []
                                    
                                    # Calculate intended moves and orthogonal slips
                                    intended_move_a = self.ACTION_TO_MOVE[action_a]
                                    intended_move_b = self.ACTION_TO_MOVE[action_b]
                                    orthogonal_moves_a = [(-intended_move_a[1], intended_move_a[0]), 
                                                          (intended_move_a[1], -intended_move_a[0])]
                                    orthogonal_moves_b = [(-intended_move_b[1], intended_move_b[0]), 
                                                          (intended_move_b[1], -intended_move_b[0])]

                                    # Consider all combinations of moves and slips
                                    move_combinations = [
                                        # No slip
                                        (intended_move_a, intended_move_b, (1 - self.slip_prob) * (1 - self.slip_prob)),
                                        # B slips, A does not
                                        (intended_move_a, orthogonal_moves_b[0], (1 - self.slip_prob) * self.slip_prob * 0.5),
                                        (intended_move_a, orthogonal_moves_b[1], (1 - self.slip_prob) * self.slip_prob * 0.5),
                                        # A slips, B does not
                                        (orthogonal_moves_a[0], intended_move_b, self.slip_prob * (1 - self.slip_prob) * 0.5),
                                        (orthogonal_moves_a[1], intended_move_b, self.slip_prob * (1 - self.slip_prob) * 0.5),
                                        # Both slip
                                        (orthogonal_moves_a[0], orthogonal_moves_b[0], self.slip_prob * self.slip_prob * 0.25),
                                        (orthogonal_moves_a[0], orthogonal_moves_b[1], self.slip_prob * self.slip_prob * 0.25),
                                        (orthogonal_moves_a[1], orthogonal_moves_b[0], self.slip_prob * self.slip_prob * 0.25),
                                        (orthogonal_moves_a[1], orthogonal_moves_b[1], self.slip_prob * self.slip_prob * 0.25),
                                    ]

                                    joint_original_action = (action_a, action_b)
                                    for move_a, move_b, move_prob in move_combinations:
                                        # if move_prob == 0:
                                        #     continue
                                        joint_move_action = (self.MOVE_TO_ACTION[move_a], self.MOVE_TO_ACTION[move_b])
                                        next_state_outcomes = self._get_next_state(state, joint_original_action, joint_move_action)
                                        for next_state_prob, next_state in next_state_outcomes:
                                            reward_a, done = self._get_reward_and_done(next_state)
                                            transitions.append((
                                                move_prob * next_state_prob,
                                                next_state,
                                                reward_a,
                                                done
                                            ))

                                    P[tuple(state.values())][(action_a, action_b)] = transitions

                                    # Assert that probabilities sum to 1
                                    total_prob = sum(t[0] for t in transitions)
                                    assert abs(total_prob - 1.0) < 1e-6, \
                                        f"Probabilities do not sum to 1 for state {state}, actions {action_a}, {action_b}. Sum: {total_prob}"

        return P


    def _get_next_state(self, state, joint_original_action, joint_move_action):
        row_a, col_a, row_b, col_b, possession = state.values()

        # terminal states
        # players in the same position are terminal states
        if row_a == row_b and col_a == col_b:
            return [(1.0, self._game_state_to_dict(state))]
        # edges of the pitch are terminal states
        if col_a == 0 or col_a == self.width - 1 or col_b == 0 or col_b == self.width - 1:
            return [(1.0, self._game_state_to_dict(state))]

        action_a, action_b = joint_original_action
        move_action_a, move_action_b = joint_move_action

        # Get potential next positions based on move actions and ball possession
        next_row_a, next_col_a = self._next_cell(row_a, col_a, move_action_a, possession == 0)
        next_row_b, next_col_b = self._next_cell(row_b, col_b, move_action_b, possession == 1)

        # Handle collisions and possession changes
        next_state_outcomes = []

        # Collision case 1: Players moving through each other
        if (row_a == row_b and 
            abs(col_a - col_b) == 1 and 
            next_col_a == col_b and 
            next_col_b == col_a) or \
           (col_a == col_b and 
            abs(row_a - row_b) == 1 and 
            next_row_a == row_b and 
            next_row_b == row_a):
            
            # Players stay in their original positions, possession changes randomly
            assert not (row_a == row_b and col_a == col_b), "Players should not be in the same cell"
            next_state_outcomes.append((0.5, self._game_state_to_dict((row_a, col_a, row_b, col_b, 0))))  # A gets possession
            next_state_outcomes.append((0.5, self._game_state_to_dict((row_a, col_a, row_b, col_b, 1))))  # B gets possession

        # Collision case 2: One player moves into the opponent's cell, the opponent stands
        elif (next_row_a == row_b and next_col_a == col_b and action_b == self.STAND) or \
             (next_row_b == row_a and next_col_b == col_a and action_a == self.STAND):
            
            # Nobody moves, they bounce back to their original location. Possession is changed.
            assert not (row_a == row_b and col_a == col_b), "Players should not be in the same cell"
            next_state_outcomes.append((1.0, self._game_state_to_dict((row_a, col_a, row_b, col_b, 1 - possession))))

        # Collision case 3: Players moving to the same cell through a bounce
        elif (row_a == next_row_a and col_a == next_col_a and action_a != self.STAND and next_row_b == row_a and next_col_b == col_a) or \
             (row_b == next_row_b and col_b == next_col_b and action_b != self.STAND and next_row_a == row_b and next_col_a == col_b):

            # Bounce back both players, random possession
            assert not (row_a == row_b and col_a == col_b), "Players should not be in the same cell"
            next_state_outcomes.append((0.5, self._game_state_to_dict((row_a, col_a, row_b, col_b, 0))))
            next_state_outcomes.append((0.5, self._game_state_to_dict((row_a, col_a, row_b, col_b, 1))))

        # Collision case 4: Players moving to the same empty cell
        elif next_row_a == next_row_b and next_col_a == next_col_b:
            assert not (row_a == next_row_b and col_a == next_col_b), "Players should not be in the same cell"
            assert not (next_row_a == row_b and next_col_a == col_b), "Players should not be in the same cell"

            # Bounce back player a, player b moves, random possession
            next_state_outcomes.append((0.25, self._game_state_to_dict((row_a, col_a, next_row_b, next_col_b, 0))))
            next_state_outcomes.append((0.25, self._game_state_to_dict((row_a, col_a, next_row_b, next_col_b, 1))))
            # Bounce back player b, player a moves, random possession
            next_state_outcomes.append((0.25, self._game_state_to_dict((next_row_a, next_col_a, row_b, col_b, 0))))
            next_state_outcomes.append((0.25, self._game_state_to_dict((next_row_a, next_col_a, row_b, col_b, 1))))
        else:
            # No collision: players move to their new positions
            assert not (next_row_a == next_row_b and next_col_a == next_col_b), "Players should not be in the same cell"
            next_state_outcomes.append((1.0, self._game_state_to_dict((next_row_a, next_col_a, next_row_b, next_col_b, possession))))

        return next_state_outcomes

    def _get_reward_and_done(self, next_state):
        row_a, col_a, row_b, col_b, possession = next_state.values()
        a_possession = possession == 0
        b_possession = possession == 1

        a_goal = a_possession and col_a == self.width - 1 and row_a in self.goal_rows
        b_goal = b_possession and col_b == 0 and row_b in self.goal_rows
        a_own_goal = a_possession and col_a == 0 and row_a in self.goal_rows
        b_own_goal = b_possession and col_b == self.width - 1 and row_b in self.goal_rows
        goal_on_a = b_goal or a_own_goal
        goal_on_b = a_goal or b_own_goal

        reward_a = 1 if goal_on_b else -1 if goal_on_a else 0
        out_of_bounds_a = col_a == 0 or col_a == self.width - 1
        out_of_bounds_b = col_b == 0 or col_b == self.width - 1
        done = out_of_bounds_a or out_of_bounds_b
        return reward_a, done

    def _next_cell(self, row, col, move_action, has_possession):
        intended_move = self.ACTION_TO_MOVE[move_action]
        new_row = max(0, min(self.height - 1, row + intended_move[1])) # Clamp to pitch height boundaries
        new_col = col + intended_move[0] # assume the move
        
        # Revert x edges unless there is a goal
        x_out_of_bounds = new_col == 0 or new_col == self.width - 1
        goal = x_out_of_bounds and new_row in self.goal_rows and has_possession
        if x_out_of_bounds and not goal:
            new_col = col  # Bounce back

        return new_row, new_col

    def step(self, joint_action):
        assert self.reset_called, "Cannot call step before calling reset"
        assert not self.episode_ended, "Episode has ended, call reset before calling step again"

        transitions = self.P[tuple(self.state.values())][tuple(joint_action)]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, next_state, reward_a, done = transitions[i]
        self.state = next_state
        self.observations = self._state_to_observations(self.state)
        self.lastaction = joint_action
        rewards = {"player_a": reward_a, "player_b": -reward_a}
        dones = {"player_a": done, "player_b": done}
        truncateds = {"player_a": False, "player_b": False}
        infos = {"player_a": {"prob": prob}, "player_b": {"prob": prob}}

        self.episode_ended = done or any(truncateds.values())

        return {"player_a": self.observations, "player_b": self.observations}, rewards, dones, truncateds, infos

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random.seed(seed)
        i = categorical_sample([is_[0] for is_ in self.isd], self.np_random)
        p, state = self.isd[i]
        self.state = self._game_state_to_dict(state)
        self.observations = self._state_to_observations(self.state)
        self.lastaction = None
        self.reset_called = True
        self.episode_ended = False
        return {"player_a": self.observations, "player_b": self.observations}, {"prob": p}

    def _action_from_direction(self, direction):
        return list(self.ACTION_TO_MOVE.keys())[list(self.ACTION_TO_MOVE.values()).index(direction)]

    def _state_to_observations(self, state):
        row_a, col_a, row_b, col_b, possession = state.values()
        n_possession_states = 2  # Explicitly define the number of possession states
        return (row_a * (self.width * self.height * self.width * n_possession_states) +
                col_a * (self.height * self.width * n_possession_states) +
                row_b * (self.width * n_possession_states) +
                col_b * n_possession_states +
                possession)

    def render(self):
        # Use self.state directly (it's already a dictionary)
        print(self.state)

        # Print player positions
        print(f"Player A position: x={self.state['col_a']}, y={self.state['row_a']}")
        print(f"Player B position: x={self.state['col_b']}, y={self.state['row_b']}")

        # Create the pitch
        print(self.width, self.height)
        pitch = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        # Add players and ball possession
        pitch[self.state['row_a']][self.state['col_a']] = 'A' + ('*' if self.state['possession'] == 0 else ' ')
        pitch[self.state['row_b']][self.state['col_b']] = 'B' + ('*' if self.state['possession'] == 1 else ' ')

        # Create a 2D array to store the entire pitch representation
        goal_start = (self.height - 1) // 2
        goal_end = goal_start + (3 if self.height % 2 else 2)

        rendered_pitch = []
        rendered_pitch.append('  ' + '-' * (self.width * 2 - 4))
        for row_index, row in enumerate(pitch):
            if row_index in range(goal_start, goal_end):
                if '*' in row[0]:
                    rendered_pitch.append(''.join(f'{cell:<2}' for cell in row[0:-1]) + '||')
                elif '*' in row[-1]:
                    rendered_pitch.append('||' + ''.join(f'{cell:<2}' for cell in row[1:]))
                else:
                    rendered_pitch.append('||' + ''.join(f'{cell:<2}' for cell in row[1:-1]) + '||')
            else:
                rendered_pitch.append(' |' + ''.join(f'{cell:<2}' for cell in row[1:-1]) + '| ')
        rendered_pitch.append('  ' + '-' * (self.width * 2 - 4))

        # Print the entire pitch
        for row in rendered_pitch:
            print(row)

        # Print additional information
        print(f"Ball possession: {'A' if self.state['possession'] == 0 else 'B'}")
        if self.lastaction:
            action_a, action_b = self.lastaction
            print(f"Last actions: A: {self.ACTION_STRING[action_a]}, B: {self.ACTION_STRING[action_b]}")
        
        # Check for goal or own goal
        if self.state['possession'] == 0:  # Player A has the ball
            if self.state['col_a'] == 0 and goal_start <= self.state['row_a'] < goal_end:
                print("OWN GOAL! Player A scored in their own goal!")
            elif self.state['col_a'] == self.width - 1 and goal_start <= self.state['row_a'] < goal_end:
                print("GOAL! Player A scored!")
        else:  # Player B has the ball
            if self.state['col_b'] == 0 and goal_start <= self.state['row_b'] < goal_end:
                print("GOAL! Player B scored!")
            elif self.state['col_b'] == self.width - 1 and goal_start <= self.state['row_b'] < goal_end:
                print("OWN GOAL! Player B scored in their own goal!")

    def _game_state_to_dict(self, game_state):
        row_a, col_a, row_b, col_b, possession = game_state
        return {
            'row_a': row_a,
            'col_a': col_a,
            'row_b': row_b,
            'col_b': col_b,
            'possession': possession
        }

    def _dict_to_game_state(self, state_dict):
        return (
            state_dict['row_a'],
            state_dict['col_a'],
            state_dict['row_b'],
            state_dict['col_b'],
            state_dict['possession']
        )

def main():
    # Create the environment
    env = SoccerSimultaneousEnv()

    # Reset the environment
    observation, info = env.reset()

    all_done = False
    n_steps = 0
    while not all_done:
        # Render the environment
        env.render()

        # Select random actions for both players
        action_a = env.action_space.sample()
        action_b = env.action_space.sample()

        # Take a step in the environment
        observation, reward, done, truncated, info = env.step((action_a, action_b))
        all_done = any(done.values()) or any(truncated.values())
        print(f"Values after step {n_steps}:")
        for player in observation.keys():
            print(f"{player}:")
            print(f"\tobservation: {observation[player]}")
            print(f"\treward: {reward[player]}")
            print(f"\tdone: {done[player]}")
            print(f"\ttruncated: {truncated[player]}")
            print(f"\tinfo: {info[player]}")

        n_steps += 1

    # Render the final state
    env.render()
    print(f"Episode finished after {n_steps} steps!")

if __name__ == "__main__":
    main()