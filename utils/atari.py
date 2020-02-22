"""Atari environment specifics

"""


def encode_action(a):
    """Map action from 0..n to game-specific semantic action values.

    Args:
        a: (int) action to encode
    Pong-specific:
        For Gym Pong n_actions == 6, which includes 4 NoOps, and 2 motion actions.
        I don't know why they did that, but we don't want to be biased to NoOps,
        so we constrain the action space to 2 or 3 actions. We will use:
            0 - noop
            2 - up
            5 - down

        See: https://github.com/openai/atari-py/blob/master/doc/manual/manual.pdf

    Example::

        >>> import gym
        >>> env= gym.make('Pong-v0')
        >>> env.action_space
        Discrete(6)
        >>> env.unwrapped.get_action_meanings()
        ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        >>> env= gym.make('Pong-v4')
        >>> env.unwrapped.get_action_meanings()
        ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    """
    assert hasattr(encode_action, 'n_actions')
    if encode_action.n_actions == 2:
        actions_encoding = {0: 2, 1: 5}
    else:
        actions_encoding = {0: 0, 1: 2, 2: 5}
    return actions_encoding[a]