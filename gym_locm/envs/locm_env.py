import gym
import numpy as np


class Card:
    def __init__(self, id, name, type, cost, attack, defense, keywords,
                 player_hp, enemy_hp, card_draw):
        self.id = id
        self.name = name
        self.type = type
        self.cost = cost
        self.attack = attack
        self.defense = defense
        self.keywords = keywords
        self.player_hp = player_hp
        self.enemy_hp = enemy_hp
        self.card_draw = card_draw

    def has_ability(self, keyword):
        return keyword in self.keywords


class LoCMEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, use_draft_history=True):
        self.state = None
        self.turn = 1

        self._cards = self._load_cards()
        self._draft = None

        cards_in_state = 33 if use_draft_history else 3
        card_features = 16

        self.cards_in_deck = 30
        self.state_shape = (cards_in_state, card_features)

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=self.state_shape,
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(3)

    def reset(self):
        self.state = np.full(self.state_shape, 0, dtype=np.float32)
        self.turn = 1

        self.draft = self._new_draft()

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass

    def _new_draft(self):
        draft = []

        for _ in range(self.cards_in_deck):
            draft.append(np.random.choice(self._cards, 3, replace=False).tolist())

        return draft

    @staticmethod
    def _load_cards():
        cards = []

        with open('gym_locm/cardlist.txt', 'r') as card_list:
            raw_cards = card_list.readlines()

            for card in raw_cards:
                cards.append(Card(*map(str.strip, card.split(';'))[:-1]))

        assert len(cards) == 160

        return cards
