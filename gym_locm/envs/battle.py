import gym
import numpy as np

from gym_locm.agents import RandomDraftAgent
from gym_locm.engine import State, Phase
from gym_locm.envs.base_env import LOCMEnv


class LOCMBattleEnv(LOCMEnv):
    metadata = {'render.modes': []}

    def __init__(self,
                 draft_agents=(RandomDraftAgent(), RandomDraftAgent()),
                 seed=None):
        super().__init__(seed=seed)

        self.draft_agents = draft_agents,

        cards_in_state = 8 + 6 + 6  # 20 cards
        card_features = 16
        player_features = 4  # hp, mana, next_rune, next_draw

        # 328 features
        self.state_shape = player_features * 2 + cards_in_state * card_features
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self.state_shape, dtype=np.float32
        )

        # 163 possible actions
        self.action_space = gym.spaces.Discrete(163)

        # reset all agents' internal state
        for agent in self.draft_agents:
            agent.reset()

        # play through draft
        while self.state.phase == Phase.DRAFT:
            for agent in self.draft_agents:
                action = agent.act(self.state)

                self.state.act(action)

    def step(self, action):
        pass  # todo: implement

    def reset(self) -> np.array:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # start a brand new game
        state = State()

        # reset all agents' internal state
        for agent in self.draft_agents:
            agent.reset()

        # play through draft
        while state.phase == Phase.DRAFT:
            for agent in self.draft_agents:
                state.act(agent.act(state))

        self.state = state

        return self._encode_state()

    def render(self, mode='human'):
        pass  # todo: implement

    @staticmethod
    def encode_players(current, opposing):
        return current.health, current.mana, current.next_rune, \
                1 + current.bonus_draw, opposing.health, \
                opposing.base_mana + opposing.bonus_mana, \
                opposing.next_rune, 1 + opposing.bonus_draw

    def _encode_state_draft(self):
        pass

    def _encode_state_battle(self):
        encoded_state = np.full(self.state_shape, 0, dtype=np.float32)

        p0, p1 = self.state.current_player, self.state.opposing_player

        # players info
        encoded_state[:8] = self.encode_players(p0, p1)

        # current player's hand and board
        encoded_state[8:8 + 16 * len(p0.hand)] = map(self.encode_card, p0.hand)
        encoded_state[136:136 + 16 * len(p0.lanes[0])] = \
            map(self.encode_card, p0.lanes[0])
        encoded_state[184:184 + 16 * len(p0.lanes[1])] = \
            map(self.encode_card, p0.lanes[1])

        # opposing player's board
        encoded_state[232:232 + 16 * len(p1.lanes[0])] = \
            map(self.encode_card, p1.lanes[0])
        encoded_state[280:280 + 16 * len(p1.lanes[1])] = \
            map(self.encode_card, p1.lanes[1])

        return encoded_state
