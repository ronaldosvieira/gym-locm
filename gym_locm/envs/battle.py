import gym
import numpy as np

from gym_locm.agents import RandomDraftAgent
from gym_locm.envs.base_env import LOCMEnv


class LOCMBattleEnv(LOCMEnv):
    metadata = {'render.modes': []}

    def __init__(self,
                 draft_agents=(RandomDraftAgent(), RandomDraftAgent()),
                 seed=None):
        super().__init__(seed=seed)

        # todo: implement the rest

    def step(self, action):
        pass  # todo: implement

    def reset(self) -> np.array:
        pass  # todo: implement

    def render(self, mode='human'):
        pass  # todo: implement

    @staticmethod
    def encode_players(current, opposing):
        pass  # todo: implement

    def _encode_state_draft(self):
        pass

    def _encode_state_battle(self):
        pass  # todo: implement
