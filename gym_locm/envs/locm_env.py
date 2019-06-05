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
    card_types = {'creature': 0, 'itemGreen': 1, 'itemRed': 2, 'itemBlue': 3}

    def __init__(self, use_draft_history=True):
        self.state = None
        self.turn = 1

        self._cards = self._load_cards()
        self._draft = None

        self.cards_in_state = 33 if use_draft_history else 3
        self.card_features = 16

        self.cards_in_deck = 30
        self.state_shape = (self.cards_in_state, self.card_features)

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=self.state_shape,
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(3)

    def reset(self):
        self.turn = 1

        self.draft = self._new_draft()
        self.state = self.draft[self.turn - 1]

        return self._convert_state()

    def step(self, action):
        pass

    def render(self, mode='human'):
        print(self._convert_state())

    def _new_draft(self):
        draft = []

        for _ in range(self.cards_in_deck):
            draft.append(np.random.choice(self._cards, 3, replace=False).tolist())

        return draft

    def _convert_state(self):
        converted_state = np.full((3, self.card_features), 0, dtype=np.float32)

        for i, card in enumerate(self.state):
            card_type = [0.0 if self.card_types[card.type] != j
                         else 1.0 for j in range(4)]
            cost = card.cost / 12
            attack = card.attack / 12
            defense = max(-12, card.defense) / 12
            keywords = list(map(int, map(lambda k: k in card.keywords,
                                         list('BCDGLW'))))
            player_hp = card.player_hp / 12
            enemy_hp = card.enemy_hp / 12
            card_draw = card.card_draw / 2

            converted_state[i] = np.array(
                card_type +
                [cost, attack, defense, player_hp, enemy_hp, card_draw] +
                keywords
            )

        return converted_state

    @staticmethod
    def _load_cards():
        cards = []

        with open('gym_locm/cardlist.txt', 'r') as card_list:
            raw_cards = card_list.readlines()

            for card in raw_cards:
                id, name, card_type, cost, attack, defense, \
                keywords, player_hp, enemy_hp, card_draw, _ = \
                    map(str.strip, card.split(';'))

                cards.append(Card(int(id), name, card_type, int(cost),
                                  int(attack), int(defense), keywords,
                                  int(player_hp), int(enemy_hp), int(card_draw)))

        assert len(cards) == 160

        return cards
