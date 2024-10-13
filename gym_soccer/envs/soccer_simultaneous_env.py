import numpy as np
from gym.envs.toy_text.utils import categorical_sample
from gym import spaces

class SoccerSimultaneousEnv:
    # Define constants for actions

    NOOP = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    ACTION_STRING = ['NOOP', 'NORTH', 'SOUTH', 'EAST', 'WEST']
    ACTION_STRING_TO_INT = {k: v for v, k in enumerate(ACTION_STRING)}

    ACTION_STRING_TO_MOVE = {
        ACTION_STRING[NOOP]: (0, 0),
        ACTION_STRING[NORTH]: (0, -1),
        ACTION_STRING[SOUTH]: (0, 1),
        ACTION_STRING[EAST]: (1, 0),
        ACTION_STRING[WEST]: (-1, 0),
    }
    MOVE_TO_ACTION_STRING = {v: k for k, v in ACTION_STRING_TO_MOVE.items()}
    ACTION_INT_TO_MOVE = {
        NOOP: (0, 0),
        NORTH: (0, -1),
        SOUTH: (0, 1),
        EAST: (1, 0),
        WEST: (-1, 0),
    }
    MOVE_TO_ACTION_INT = {v: k for k, v in ACTION_INT_TO_MOVE.items()}


    def __init__(self, width=5, height=4, slip_prob=0.0, player_a_policy=None, player_b_policy=None):

        # Assert that both policies cannot be set simultaneously
        assert not (player_a_policy and player_b_policy), "Both players cannot have a policy. At least one must be None."
        if player_a_policy is not None:
            assert isinstance(player_a_policy, dict), "Player A policy must be a dictionary."
        if player_b_policy is not None:
            assert isinstance(player_b_policy, dict), "Player B policy must be a dictionary."

        # Minimum pitch size is 5x4
        assert width >= 5, "Width must be at least 5 columns."
        assert height >= 4, "Height must be at least 4 rows."

        self.width = width + 2  # +2 for the columns where goals are located
        self.height = height
        self.slip_prob = slip_prob
        self.player_a_policy = player_a_policy
        self.player_b_policy = player_b_policy
        self.multiagent = player_a_policy is None and player_b_policy is None
        self.np_random = np.random.RandomState()

        self.goal_rows = ((self.height - 1) // 2, self.height // 2) if self.height % 2 == 0 else (self.height // 2 - 1, self.height // 2, self.height // 2 + 1)
        self.goal_cols = (0, self.width - 1)

        self.unreachable_states, self.terminal_states = [], {} # containing rewards for player A
        self.state_space, self.n_states = {}, 1 # single terminal state
        for xa in range(self.height):
            for ya in range(self.width):
                for xb in range(self.height):
                    for yb in range(self.width):
                        for p in range(2):
                            state_tuple = (xa, ya, xb, yb, p)

                            # Top/bottom left/right corners (goal columns but not goal)
                            if ya in self.goal_cols and xa not in self.goal_rows or \
                               yb in self.goal_cols and xb not in self.goal_rows:
                                self.unreachable_states.append(state_tuple)
                                continue

                            # Goals without possession
                            if xa in self.goal_rows and ya in self.goal_cols and p != 0 or \
                               xb in self.goal_rows and yb in self.goal_cols and p != 1:
                                self.unreachable_states.append(state_tuple)
                                continue

                            # Players occupy the same cell
                            if xa == xb and ya == yb:
                                self.unreachable_states.append(state_tuple)
                                continue

                            self.state_space[state_tuple] = self.n_states
                            self.n_states += 1

                            # Terminal states, goals (with possession)
                            if xa in self.goal_rows and ya in self.goal_cols and p == 0 or \
                               xb in self.goal_rows and yb in self.goal_cols and p == 1:
                                # Goal for player A, or player B own goal
                                ga = p == 0 and xa in self.goal_rows and ya == self.width - 1 or \
                                    p == 1 and xb in self.goal_rows and yb == self.width - 1
                                # Goal for player B, or player A own goal
                                gb = p == 1 and xb in self.goal_rows and yb == 0 or \
                                    p == 0 and xa in self.goal_rows and ya == 0

                                assert ga or gb, "At least one goal must have been scored to be here"
                                assert not (ga and gb), "We cannot have both goals scored"
                                self.terminal_states[state_tuple] = 1.0 if ga else -1.0 if gb else 0.0
                                self.n_states -= 1 # only one terminal state is included in the observation space

        # Initialize the state space and action space
        # self.n_states = self.width * self.height * self.width * self.height * 2  # width * height * width * height * 2 (possession)
        self.n_actions = len(self.ACTION_STRING)  # Actions: UP, DOWN, LEFT, RIGHT, STAND

        # # TODO: this is a test, remove it
        # # Generate a random policy for player B
        # import pickle
        # self.player_b_policy = {}
        # for s in range(self.n_states):
        #     self.player_b_policy[s] = self.np_random.randint(0, self.n_actions)
        # self.multiagent = False
        # # Save dictionary to a file
        # with open('random_policy_5x4.pkl', 'wb') as f:
        #     pickle.dump(self.player_b_policy, f)

        # Update observation space to be Discrete
        self.observation_space = spaces.Tuple((spaces.Discrete(self.n_states), spaces.Discrete(self.n_states))) if self.multiagent else spaces.Discrete(self.n_states)
        self.action_space = spaces.Tuple((spaces.Discrete(self.n_actions), spaces.Discrete(self.n_actions))) if self.multiagent else spaces.Discrete(self.n_actions)

        # Define the initial state distribution
        self.isd = self._generate_isd()

        # Define transition dynamics and create observation cache
        self.P, self.P_readable = self._initialize_transition_dynamics()

        # Add a flag to track if reset has been called
        self.needs_reset = True

        # Initialize self.state and self.observations
        self.state = None
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
        P_readable = {}

        for xa in range(self.height):
            for ya in range(self.width):
                for xb in range(self.height):
                    for yb in range(self.width):
                        for p in range(2):  # 0: A, 1: B
                            st = (xa, ya, xb, yb, p)
                            if st in self.unreachable_states:
                                continue # skip unreachable states

                            s = self.state_space[st]
                            P[s] = {}
                            P_readable[st] = {}

                            # All actions integer for a and b, sample a policy if provided
                            aaa = list(range(self.n_actions)) if self.player_a_policy is None else [self.player_a_policy[s]]
                            aab = list(range(self.n_actions)) if self.player_b_policy is None else [self.player_b_policy[s]]
                            for aa in aaa:
                                asa = self.ACTION_STRING[aa]

                                for ab in aab:
                                    asb = self.ACTION_STRING[ab]

                                    # Original joint action, integer and string
                                    ja = (aa, ab)
                                    jas = (asa, asb)

                                    transitions = []
                                    transitions_readable = []

                                    # Calculate intended moves for a and b, as well as orthogonal slips
                                    ma = self.ACTION_INT_TO_MOVE[aa]
                                    mb = self.ACTION_INT_TO_MOVE[ab]
                                    mas = [(-ma[1], ma[0]), (ma[1], -ma[0])]
                                    mbs = [(-mb[1], mb[0]), (mb[1], -mb[0])]

                                    # All move combinations to consider
                                    amc = [
                                        # No slip
                                        (ma, mb, (1 - self.slip_prob) * (1 - self.slip_prob)),
                                        # B slips, A does not
                                        (ma, mbs[0], (1 - self.slip_prob) * self.slip_prob * 0.5),
                                        (ma, mbs[1], (1 - self.slip_prob) * self.slip_prob * 0.5),
                                        # A slips, B does not
                                        (mas[0], mb, self.slip_prob * (1 - self.slip_prob) * 0.5),
                                        (mas[1], mb, self.slip_prob * (1 - self.slip_prob) * 0.5),
                                        # Both slip
                                        (mas[0], mbs[0], self.slip_prob * self.slip_prob * 0.25),
                                        (mas[0], mbs[1], self.slip_prob * self.slip_prob * 0.25),
                                        (mas[1], mbs[0], self.slip_prob * self.slip_prob * 0.25),
                                        (mas[1], mbs[1], self.slip_prob * self.slip_prob * 0.25),
                                    ]

                                    for ma, mb, mp in amc:
                                        if mp == 0:
                                            continue # remove zero probability transitions

                                        # Joint move action
                                        jma = (ma, mb)

                                        # Get all next state possible outcomes for the action, and move (slip)
                                        nso = self._get_next_state(st, ja, jma)
                                        for nsp, ns in nso:
                                            d = ns in self.terminal_states
                                            r = self.terminal_states[ns] if d else 0.0
                                            transitions.append((
                                                mp * nsp, # probability of the move (slip), and next_state
                                                self.state_space[ns], # next state
                                                r, # reward
                                                d # done
                                            ))
                                            transitions_readable.append((
                                                mp * nsp, # probability of the move (slip), and next_state
                                                ns, # next state
                                                r, # reward
                                                d # done
                                            ))
                                    # if we need to account for joint actions
                                    if self.multiagent:
                                        P[s][ja] = transitions
                                        P_readable[st][jas] = transitions_readable
                                    # if we need to account for individual actions a and b
                                    elif self.player_a_policy is None and self.player_b_policy is not None:
                                        P[s][aa] = transitions
                                        P_readable[st][asa] = transitions_readable
                                    elif self.player_b_policy is None and self.player_a_policy is not None:
                                        P[s][ab] = transitions
                                        P_readable[st][asb] = transitions_readable
                                    # error case
                                    else:
                                        raise ValueError("No policy provided for both players, but action is an integer")

                                    # Assert that probabilities sum to 1
                                    tp = sum(t[0] for t in transitions)
                                    assert abs(tp - 1.0) < 1e-6, \
                                        f"Probabilities do not sum to 1 for state {st}, actions {aa}, {ab}. Sum: {tp}"

        return P, P_readable


    def _get_next_state(self, st, ja, jma):
        xa, ya, xb, yb, p = st

        # terminal states
        if st in self.terminal_states:
            return [(1.0, st)]

        # original action integers and move action (including slips)
        aa, ab = ja
        maa, mab = jma

        # Get potential next positions based on move actions and ball possession
        nxa, nya = self._next_cell(xa, ya, maa, p == 0)
        nxb, nyb = self._next_cell(xb, yb, mab, p == 1)

        # Handle collisions and possession changes
        nso = []

        # Collision case 1: Players moving through each other
        if (xa == xb and
            abs(ya - yb) == 1 and
            nya == yb and
            nyb == ya) or \
           (ya == yb and
            abs(xa - xb) == 1 and
            nxa == xb and
            nxb == xa):

            # Players stay in their original positions, possession changes randomly
            assert not (xa == xb and ya == yb), "Players should not be in the same cell"
            nso.append((0.5, (xa, ya, xb, yb, 0)))  # A gets possession
            nso.append((0.5, (xa, ya, xb, yb, 1)))  # B gets possession

        # Collision case 2: One player moves into the opponent's cell, the opponent stands
        elif (nxa == xb and nya == yb and ab == self.NOOP) or \
             (nxb == xa and nyb == ya and aa == self.NOOP):

            # Nobody moves, they bounce back to their original location. Possession is changed.
            assert not (xa == xb and ya == yb), "Players should not be in the same cell"
            nso.append((1.0, (xa, ya, xb, yb, 1 - p)))

        # Collision case 3: Players moving to the same cell through a bounce
        elif (xa == nxa and ya == nya and aa != self.NOOP and nxb == xa and nyb == ya) or \
             (xb == nxb and yb == nyb and ab != self.NOOP and nxa == xb and nya == yb):

            # Bounce back both players, random possession
            assert not (xa == xb and ya == yb), "Players should not be in the same cell"
            nso.append((0.5, (xa, ya, xb, yb, 0)))
            nso.append((0.5, (xa, ya, xb, yb, 1)))

        # Collision case 4: Players moving to the same empty cell
        elif nxa == nxb and nya == nyb:
            assert not (xa == nxb and ya == nyb), "Players should not be in the same cell"
            assert not (nxa == xb and nya == yb), "Players should not be in the same cell"

            # Bounce back player a, player b moves, random possession
            nso.append((0.25, (xa, ya, nxb, nyb, 0)))
            nso.append((0.25, (xa, ya, nxb, nyb, 1)))
            # Bounce back player b, player a moves, random possession
            nso.append((0.25, (nxa, nya, xb, yb, 0)))
            nso.append((0.25, (nxa, nya, xb, yb, 1)))
        else:
            # No collision: players move to their new positions
            assert not (nxa == nxb and nya == nyb), "Players should not be in the same cell"
            nso.append((1.0, (nxa, nya, nxb, nyb, p)))

        return nso

    def _next_cell(self, x, y, ma, p):
        nx = max(0, min(self.height - 1, x + ma[1])) # Clamp to pitch height boundaries
        ny = y + ma[0] # assume the move in y

        # Revert x edges unless there is a goal (and not out of bounds)
        xoob = ny == 0 or ny == self.width - 1
        goal = xoob and nx in self.goal_rows and p # has possession
        if xoob and not goal:
            ny = y  # Bounce back
        return nx, ny

    def step(self, action):
        assert not self.needs_reset, "Please reset the environment before taking a step"

        if self.multiagent:
            assert (isinstance(action, tuple) and len(action) == 2), "Action must be a tuple of length 2 for multiagent case"
        else:
            assert isinstance(action, int), "Action must be an integer for single agent case"

        action_readable = self.ACTION_STRING[action] if not self.multiagent else (self.ACTION_STRING[action[0]], self.ACTION_STRING[action[1]])
        transitions = self.P_readable[self.state][action_readable]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, self.state, reward, done = transitions[i]
        self.observations = (self.state_space[self.state], self.state_space[self.state]) if self.multiagent else self.state_space[self.state]
        self.lastaction = action
        rewards = (reward, -reward) if self.multiagent else reward
        dones = (done, done) if self.multiagent else done
        truncateds = (False, False) if self.multiagent else False
        infos = ({"p": prob}, {"p": prob}) if self.multiagent else {"p": prob}
        self.needs_reset = done or (any(truncateds) if self.multiagent else truncateds)

        return self.observations, rewards, dones, truncateds, infos

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random.seed(seed)

        i = categorical_sample([is_[0] for is_ in self.isd], self.np_random)
        p, self.state = self.isd[i]
        # currently the integer representation of the state
        # later we need the rotation, then integer (both player "see" the same perspective)
        # also this observation is the same integer for both, later it won't
        self.observations = (self.state_space[self.state], self.state_space[self.state]) if self.multiagent else self.state_space[self.state]
        infos = ({"p": p}, {"p": p}) if self.multiagent else {"p": p}
        self.lastaction = None
        self.needs_reset = False
        return self.observations, infos

    def render(self):
        # Use self.state directly (it's already a dictionary)
        print(self.state)
        xa, ya, xb, yb, p = self.state

        # Print player positions
        print(f"Player A position: x={xa}, y={ya}, possession={p==0}")
        print(f"Player B position: x={xb}, y={yb}, possession={p==1}")

        # Create the pitch
        pitch = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        # Add players and ball possession
        pitch[xa][ya] = 'A' + ('*' if p == 0 else ' ')
        pitch[xb][yb] = 'B' + ('*' if p == 1 else ' ')

        # Create a 2D array to store the entire pitch representation
        goal_start = (self.height - 1) // 2
        goal_end = goal_start + (3 if self.height % 2 else 2)

        rendered_pitch = []
        rendered_pitch.append('  ' + '-' * (self.width * 2 - 4))
        for ri, r in enumerate(pitch):
            if ri in range(goal_start, goal_end):
                if '*' in r[0]:
                    rendered_pitch.append(''.join(f'{cell:<2}' for cell in r[0:-1]) + '||')
                elif '*' in r[-1]:
                    rendered_pitch.append('||' + ''.join(f'{cell:<2}' for cell in r[1:]))
                else:
                    rendered_pitch.append('||' + ''.join(f'{cell:<2}' for cell in r[1:-1]) + '||')
            else:
                rendered_pitch.append(' |' + ''.join(f'{cell:<2}' for cell in r[1:-1]) + '| ')
        rendered_pitch.append('  ' + '-' * (self.width * 2 - 4))

        # Print the entire pitch
        for r in rendered_pitch:
            print(r)

        # Print additional information
        print(f"Ball possession: {'A' if p == 0 else 'B'}")
        if self.lastaction and self.multiagent:
            action_a, action_b = self.lastaction
            print(f"Last actions: A: {self.ACTION_STRING[action_a]}, B: {self.ACTION_STRING[action_b]}")
        elif self.lastaction and not self.multiagent:
            if self.player_a_policy is None:
                action_a = self.lastaction
                print(f"Last action: A: {self.ACTION_STRING[action_a]}")
            elif self.player_b_policy is None:
                action_b = self.lastaction
                print(f"Last action: B: {self.ACTION_STRING[action_b]}")
            else:
                raise ValueError("No policy provided for both players, but action is an integer")

        # Check for goal or own goal
        if p == 0:  # Player A has the ball
            if ya == 0 and goal_start <= xa < goal_end:
                print("OWN GOAL! Player A scored in their own goal!")
            elif ya == self.width - 1 and goal_start <= xa < goal_end:
                print("GOAL! Player A scored!")
        else:  # Player B has the ball
            if yb == 0 and goal_start <= xb < goal_end:
                print("GOAL! Player B scored!")
            elif yb == self.width - 1 and goal_start <= xb < goal_end:
                print("OWN GOAL! Player B scored in their own goal!")

def main():
    import pickle
    with open('random_policy_5x4.pkl', 'rb') as f:
        random_policy = pickle.load(f)

    # Create the environment
    env = SoccerSimultaneousEnv(width=5, height=4, slip_prob=0.2, player_a_policy=None, player_b_policy=random_policy)

    # Reset the environment
    os, fs = env.reset()

    all_done = False
    n_steps = 0
    while not all_done:
        # Render the environment
        env.render()

        # Select random actions for both players
        action_a = env.action_space.sample()
        action_b = env.action_space.sample()

        # Take a step in the environment
        # observation, reward, done, truncated, info = env.step((action_a, action_b))
        os, rs, ds, ts, fs = env.step(action_a)
        all_done = any(ds) or any(ts)
        print(f"Values after step {n_steps}:")
        for i, po in enumerate(os):
            print(f"{po}:")
            print(f"\tobservation: {os[i]}")
            print(f"\treward: {rs[i]}")
            print(f"\tdone: {ds[i]}")
            print(f"\ttruncated: {ts[i]}")
            print(f"\tinfo: {fs[i]}")

        n_steps += 1

    # Render the final state
    env.render()
    print(f"Episode finished after {n_steps} steps!")

if __name__ == "__main__":
    main()