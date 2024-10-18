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
    TERMINAL_STATE = (-1, -1, -1, -1, -1)


    def __init__(self, width=5, height=4, slip_prob=0.0, player_a_policy=None, player_b_policy=None, seed=0):

        # Assert that both policies cannot be set simultaneously
        assert not (player_a_policy is not None and player_b_policy is not None), "Both players cannot have a policy. At least one must be None."
        # if player_a_policy is not None:
        #     assert isinstance(player_a_policy, dict), "Player A policy must be a dictionary."
        # if player_b_policy is not None:
        #     assert isinstance(player_b_policy, dict), "Player B policy must be a dictionary."

        # Minimum pitch size is 5x4
        assert width >= 5, "Width must be at least 5 columns."
        assert height >= 4, "Height must be at least 4 rows."

        self.width = width + 2  # +2 for the columns where goals are located
        self.height = height
        self.slip_prob = slip_prob
        self.seed = seed
        self.player_a_policy = player_a_policy
        self.player_b_policy = player_b_policy
        self.multiagent = player_a_policy is None and player_b_policy is None
        self.return_agent = ['player_a', 'player_b'] if self.multiagent else ['player_a'] \
            if player_a_policy is None else ['player_b']
        self.np_random = np.random.RandomState()
        self.np_random.seed(self.seed)

        self.goal_rows = ((self.height - 1) // 2, self.height // 2) if self.height % 2 == 0 else (self.height // 2 - 1, self.height // 2, self.height // 2 + 1)
        self.goal_cols = (0, self.width - 1)

        self.unreachable_states, self.goal_states = [], {} # containing rewards for player A
        self.state_space, self.nS = {}, 1
        self.state_space[self.TERMINAL_STATE] = 0 # initialize the terminal state
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
                                self.goal_states[state_tuple] = 1.0 if ga else -1.0 if gb else 0.0
                                continue

                            self.state_space[state_tuple] = self.nS
                            self.nS += 1

        assert self.nS == len(self.state_space), "State space should be the same length as the number of states"
        self._reverse_state_space = {v: k for k, v in self.state_space.items()}
        # Initialize the state space and action space
        # self.n_states = self.width * self.height * self.width * self.height * 2  # width * height * width * height * 2 (possession)
        self.nA = len(self.ACTION_STRING)  # Actions: UP, DOWN, LEFT, RIGHT, STAND

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
        self.observation_space = spaces.Dict({
            a: spaces.Discrete(self.nS) for a in self.return_agent
        })
        self.action_space = spaces.Dict({
            a: spaces.Discrete(self.nA) for a in self.return_agent
        })

        # Define the initial state distribution
        self.isd = self._generate_isd()

        # Define transition dynamics and create observation cache
        self.P, self.P_readable, self.Pmat, self.Rmat = self._initialize_transition_dynamics()

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
        Pmat = np.zeros([self.nS, self.nS, self.nA, self.nA]) if self.multiagent else np.zeros([self.nS, self.nS, self.nA])
        Rmat = np.zeros([self.nS, self.nA, self.nA]) if self.multiagent else np.zeros([self.nS, self.nA])

        for xa in range(self.height):
            for ya in range(self.width):
                for xb in range(self.height):
                    for yb in range(self.width):
                        for p in range(2):  # 0: A, 1: B
                            st = (xa, ya, xb, yb, p)
                            if st in self.unreachable_states:
                                continue # skip unreachable states

                            s = self._state_to_observation(st)
                            P[s] = {}
                            P_readable[st] = {}

                            # All actions integer for a and b, sample a policy if provided
                            aaa = list(range(self.nA)) if self.player_a_policy is None else [self.player_a_policy[s]]
                            aab = list(range(self.nA)) if self.player_b_policy is None else [self.player_b_policy[s]]
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
                                            if st == ns and st in self.goal_states:
                                                d, r = True, 0.0
                                            elif st != ns and ns in self.goal_states:
                                                d, r = True, self.goal_states[ns]
                                            else:
                                                d, r = False, 0.0
                                            p = mp * nsp
                                            # flip reward for player B in single agent case
                                            if not self.multiagent and 'player_b' in self.return_agent:
                                                r = -1 * r
                                            transitions.append((
                                                p, # probability of the move (slip), and next_state
                                                self._state_to_observation(ns), # next state
                                                r, # reward
                                                d # done
                                            ))
                                            transitions_readable.append((
                                                p, # probability of the move (slip), and next_state
                                                ns, # next state
                                                r, # reward
                                                d # done
                                            ))
                                    # if we need to account for joint actions
                                    if self.multiagent:
                                        P[s][ja] = transitions
                                        Rmat[s][ja[0]][ja[1]] = 0  # Initialize reward to 0
                                        for prob, next_state, reward, done in transitions:
                                            Pmat[s][next_state][ja[0]][ja[1]] += prob
                                            Rmat[s][ja[0]][ja[1]] += prob * reward  # Weighted sum of rewards
                                        P_readable[st][jas] = transitions_readable
                                    # if we need to account for individual actions a and b
                                    elif self.player_a_policy is None and self.player_b_policy is not None:
                                        P[s][aa] = transitions
                                        Rmat[s][aa] = 0  # Initialize reward to 0
                                        for prob, next_state, reward, done in transitions:
                                            Pmat[s][next_state][aa] += prob
                                            Rmat[s][aa] += prob * reward  # Weighted sum of rewards
                                        P_readable[st][asa] = transitions_readable
                                    elif self.player_b_policy is None and self.player_a_policy is not None:
                                        P[s][ab] = transitions
                                        Rmat[s][ab] = 0  # Initialize reward to 0
                                        for prob, next_state, reward, done in transitions:
                                            Pmat[s][next_state][ab] += prob
                                            Rmat[s][ab] += prob * reward  # Weighted sum of rewards
                                        P_readable[st][asb] = transitions_readable
                                    # error case
                                    else:
                                        raise ValueError("No policy provided for both players, but action is an integer")

                                    # Assert that probabilities sum to 1
                                    tp = sum(t[0] for t in transitions)
                                    assert abs(tp - 1.0) < 1e-6, \
                                        f"Probabilities do not sum to 1 for state {st}, actions {aa}, {ab}. Sum: {tp}"

        # P is the compact representation of the transition dynamics
        # P_readable is the same but with the states represented as tuples
        # P_readable terminal states are the tuples that are in goal_states
        # P has 0 as the terminal states
        return P, P_readable, Pmat, Rmat


    def _get_next_state(self, st, ja, jma):
        xa, ya, xb, yb, p = st

        # terminal states
        if st in self.goal_states:
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
        assert isinstance(action, dict), "Action must be a dictionary"
        assert len(action) == 1 or len(action) == 2, "Action must be a dictionary of length 1 or 2"
        assert self.multiagent or self.player_a_policy is not None or self.player_b_policy is not None, "Multiagent environment or policy for one player must be provided"
        assert self.player_a_policy is not None or 'player_a' in action, "A policy for player_a must be provided"
        assert self.player_b_policy is not None or 'player_b' in action, "A policy for player_b must be provided"

        only_agent = None
        if self.multiagent:
            assert (isinstance(action, dict) and len(action) == 2), "Action must be a dictionary of length 2 for multiagent case"
            assert 'player_a' in action and 'player_b' in action, "Action must contain both 'player_a' and 'player_b'"
        else:
            assert (isinstance(action, dict) and len(action) == 1), "Action must be a dictionary of length 1 for single agent case"
            assert 'player_a' in action or 'player_b' in action, "Action must contain either 'player_a' or 'player_b'"
            assert not ('player_a' in action and 'player_b' in action), "Action must contain only one of 'player_a' or 'player_b'"
            only_agent = 'player_a' if self.player_a_policy is None else 'player_b'

        action_readable = (self.ACTION_STRING[action['player_a']], self.ACTION_STRING[action['player_b']]) if self.multiagent else self.ACTION_STRING[action[only_agent]]
        transitions = self.P_readable[self.state][action_readable]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, self.state, reward, done = transitions[i]
        self.observations = {a: self._state_to_observation(self.state) for a in self.return_agent}
        self.lastaction = action
        self.timestep += 1
        rewards = {a: reward for a in self.return_agent}
        if self.multiagent:
            rewards['player_b'] *= -1
        dones = {a: done for a in self.return_agent}
        truncateds = {a: self.timestep >= 100 for a in self.return_agent}
        infos = {a: {"p": np.round(prob, 2)} for a in self.return_agent}
        self.needs_reset = any(dones.values()) or any(truncateds.values())

        return self.observations, rewards, dones, truncateds, infos

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random.seed(seed)

        i = categorical_sample([is_[0] for is_ in self.isd], self.np_random)
        p, self.state = self.isd[i]
        # currently the integer representation of the state
        # later we need the rotation, then integer (both player "see" the same perspective)
        # also this observation is the same integer for both, later it won't
        self.observations = {a: self._state_to_observation(self.state) for a in self.return_agent}
        infos = {a: {"p": np.round(p, 2)} for a in self.return_agent}
        self.lastaction = None
        self.needs_reset = False
        self.timestep = 0
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

        rendered_pitch = []
        rendered_pitch.append('  ' + '-' * (self.width * 2 - 4))
        for ri, r in enumerate(pitch):
            if ri in self.goal_rows:
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
            action_a, action_b = self.lastaction.values()
            print(f"Last actions: A: {self.ACTION_STRING[action_a]}, B: {self.ACTION_STRING[action_b]}")
        elif self.lastaction and not self.multiagent:
            if self.player_a_policy is None:
                action_a = self.lastaction['player_a']
                print(f"Last action: A: {self.ACTION_STRING[action_a]}")
            elif self.player_b_policy is None:
                action_b = self.lastaction['player_b']
                print(f"Last action: B: {self.ACTION_STRING[action_b]}")
            else:
                raise ValueError("No policy provided for both players, but action is an integer")

        # Check for goal or own goal
        if p == 0:  # Player A has the ball
            if ya == 0 and xa in self.goal_rows:
                print("OWN GOAL! Player A scored in their own goal!")
            elif ya == self.width - 1 and xa in self.goal_rows:
                print("GOAL! Player A scored!")
        else:  # Player B has the ball
            if yb == 0 and xb in self.goal_rows:
                print("GOAL! Player B scored!")
            elif yb == self.width - 1 and xb in self.goal_rows:
                print("OWN GOAL! Player B scored in their own goal!")

    def _state_to_observation(self, state):
        # This function later should rotate the observations
        # so that both players see the same perspective
        # currently it's they see the global game state
        # the problem is the a players trained to solve player a's perspective
        # cannot perform on player b's perspective
        state = self.TERMINAL_STATE if state in self.goal_states else state
        return self.state_space[state]

    def _observation_to_state(self, observation):
        return self._reverse_state_space[observation]

def main():
    n_states = 761 # 5x4 field
    # n_states = 11705 # 11x7 field
    n_actions = 5
    import time
    from gym_soccer.utils.policies import get_random_policy, get_stand_policy
    from gym_soccer.utils.planners import value_iteration, policy_iteration, modified_policy_iteration

    random_policy = get_random_policy(n_states, n_actions, seed=0)
    stand_policy = get_stand_policy(n_states)
    player_b_policy = random_policy

    # Create the environment
    # env = SoccerSimultaneousEnv(
    #     width=5, height=4, slip_prob=0.2,
    #     player_a_policy=None, player_b_policy=None)
    # env = SoccerSimultaneousEnv(
    #     width=11, height=7, slip_prob=0.2,
    #     player_a_policy=None, player_b_policy=player_b_policy)
    env = SoccerSimultaneousEnv(
        width=5, height=4, slip_prob=0.2,
        player_a_policy=None, player_b_policy=player_b_policy)
    # env = SoccerSimultaneousEnv(
    #     width=5, height=4, slip_prob=0.2,
    #     player_a_policy=player_b_policy, player_b_policy=None)

    k_1 = 1
    k_2 = 10000000
    theta = 1e-10
    discount_factor = 0.99
    # Value iteration
    vi_time = time.time()
    vi_br_pi, vi_br_V, vi_br_Q, vi_cc = value_iteration(env, theta=theta, discount_factor=discount_factor)
    vi_time = time.time() - vi_time
    print("Value iteration converged in {} iterations in {:.2f} seconds".format(vi_cc, vi_time))

    # Policy iteration
    pi_time = time.time()
    pi_br_pi, pi_br_V, pi_br_Q, pi_cc = policy_iteration(env, theta=theta, discount_factor=discount_factor)
    pi_time = time.time() - pi_time
    print("Policy iteration converged in {} iterations in {:.2f} seconds".format(pi_cc, pi_time))

    # Modified policy iteration, 1 pass for each policy evaluation
    mpi_1_time = time.time()
    mpi_1_br_pi, mpi_1_br_V, mpi_1_br_Q, mpi_1_cc = modified_policy_iteration(env, k=k_1, theta=theta, discount_factor=discount_factor)
    mpi_1_time = time.time() - mpi_1_time
    print("Modified policy iteration (k={}) converged in {} iterations in {:.2f} seconds".format(k_1, mpi_1_cc, mpi_1_time))

    # Modified policy iteration, infinite passes for each policy evaluation
    mpi_2_time = time.time()
    mpi_2_br_pi, mpi_2_br_V, mpi_2_br_Q, mpi_2_cc = modified_policy_iteration(env, k=k_2, theta=theta, discount_factor=discount_factor)
    mpi_2_time = time.time() - mpi_2_time
    print("Modified policy iteration (k={}) converged in {} iterations in {:.2f} seconds".format(k_2, mpi_2_cc, mpi_2_time))

    # Check if all policies are the same
    assert np.all(vi_br_pi == pi_br_pi), "Value iteration and policy iteration should converge to the same policy"
    assert np.all(vi_br_pi == mpi_1_br_pi), "Value iteration and modified policy iteration should converge to the same policy"
    assert np.all(vi_br_pi == mpi_2_br_pi), "Value iteration and modified policy iteration should converge to the same policy"

    # Check if all value functions are the same
    assert np.allclose(vi_br_V, pi_br_V), "Value iteration and policy iteration should converge to the same value function"
    assert np.allclose(vi_br_V, mpi_1_br_V), "Value iteration and modified policy iteration should converge to the same value function"
    assert np.allclose(vi_br_V, mpi_2_br_V), "Value iteration and modified policy iteration should converge to the same value function"

    # Check if all Q-functions are the same
    assert np.allclose(vi_br_Q, pi_br_Q), "Value iteration and policy iteration should converge to the same Q-function"
    assert np.allclose(vi_br_Q, mpi_1_br_Q), "Value iteration and modified policy iteration should converge to the same Q-function"
    assert np.allclose(vi_br_Q, mpi_2_br_Q), "Value iteration and modified policy iteration should converge to the same Q-function"
    print("All algorithms converged to the same result.")

    n_episodes = 1000
    rewards, steps = [], []
    for i in range(n_episodes):

        # Reset the environment
        os, fs = env.reset()
        rewards.append(0)
        steps.append(0)
        all_done = False
        while not all_done:

            # Render the environment
            if i == n_episodes - 1:
                env.render()

            # Select random actions for both players
            # action_a = env.action_space['player_a'].sample()
            # action_b = env.action_space.sample()
            action_a = vi_br_pi[os['player_a']]
            # action_a = vi_br_pi[os['player_b']]
            # action_a = env.EAST

            # Take a step in the environment
            # observation, reward, done, truncated, info = env.step({'player_a': action_a, 'player_b': action_b})
            os, rs, ds, ts, fs = env.step({'player_a': action_a})
            rewards[-1] += rs['player_a']

            all_done = any(ds.values()) or any(ts.values())
            if i == n_episodes - 1:
                print(f"Values after step {steps[-1]}:")
                for k, po in os.items():
                    print(f"{po}:")
                    print(f"\tobservation: {os[k]}")
                    print(f"\treward: {rs[k]}")
                    print(f"\tdone: {ds[k]}")
                    print(f"\ttruncated: {ts[k]}")
                    print(f"\tinfo: {fs[k]}")

            steps[-1] += 1

        if i == n_episodes - 1:
            # Render the final state
            env.render()

    print(f"All {n_episodes} episodes finished with average reward {np.mean(rewards)} and average steps {np.mean(steps)}.")


if __name__ == "__main__":
    main()