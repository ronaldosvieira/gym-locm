import gym
import numpy as np


class LoCMEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, use_draft_history=True):
        self.state = None
        self.turn = 1

        cards_in_state = 33 if use_draft_history else 3
        card_features = 16

        self.state_shape = (cards_in_state, card_features)

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=self.state_shape,
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(3)

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass
