import gym
import numpy as np

from gym_locm.engine import Phase
from gym_locm.envs.base_env import LOCMEnv


class LOCMFullGameEnv(LOCMEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

        self.choices = ([], [])

        cards_in_draft_state = 3
        cards_in_battle_state = 8 + 6 + 6

        player_features = 4  # hp, mana, next_rune, next_draw
        card_features = 16

        self.state_shapes = {
            Phase.DRAFT: (cards_in_draft_state, card_features),
            Phase.BATTLE: (player_features * 2 + cards_in_battle_state * card_features)
        }

        self.observation_spaces = {
            Phase.DRAFT: gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32,
                                        shape=self.state_shapes[Phase.DRAFT]),
            Phase.BATTLE: gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32,
                                         shape=self.state_shapes[Phase.BATTLE])
        }

        self.action_spaces = {
            Phase.DRAFT: gym.spaces.Discrete(3),
            Phase.BATTLE: gym.spaces.Discrete(163)
        }

    @property
    def observation_space(self):
        return self.observation_spaces[self.state.phase]

    @property
    def action_space(self):
        return self.action_spaces[self.state.phase]

    def reset(self):
        pass

    def step(self, action):
        pass

    def _encode_state_battle(self):
        pass

    def _encode_state_draft(self):
        pass
