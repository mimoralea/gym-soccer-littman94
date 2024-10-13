from gym.envs.registration import register


# classics
register(
    id='Soccer5x4Simultaneous-v0',
    entry_point='gym_soccer.envs:SoccerEnv',
    kwargs={'width': 5, 'height': 4, 'slip_prob': 0.2, 'isd_possession_a': 0.5, 'player_a_policy': None, 'player_b_policy': None},
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True,
)
