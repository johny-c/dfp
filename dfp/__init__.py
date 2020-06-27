from collections import namedtuple


Transition = namedtuple('Transition', ['obs', 'action', 'reward', 'next_obs', 'terminal'])
