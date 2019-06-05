from typing import List

import gym
import numpy as np

from enum import Enum, IntEnum


class Phase(Enum):
    DRAFT = 0
    BATTLE = 1


class PlayerOrder(IntEnum):
    FIRST = 0
    SECOND = 1


class Lane(IntEnum):
    LEFT = 0
    RIGHT = 1


class Player:
    def __init__(self):
        self.health = 30
        self.base_mana = 1
        self.mana = self.base_mana
        self.next_rune = 25
        self.draw = 1

        self.deck = []
        self.hand = []


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


class GameState:
    def __init__(self, current_player, players, lanes):
        self.current_player = current_player
        self.players = players
        self.lanes = lanes


class Action:
    pass


class DraftAction(Action):
    def __init__(self, chosen_card_index):
        self.chosen_card_index = chosen_card_index


class BattleAction(Action):
    pass  # todo: implement


class Game:
    _draft_cards: List[List[Card]]
    current_player: PlayerOrder
    current_phase: Phase

    def __init__(self, cards_in_deck=30):
        self.cards_in_deck = cards_in_deck

        self._cards = self._load_cards()
        self.players = []
        self.lanes = []
        self.turn = -1

        self.reset()

    def reset(self) -> GameState:
        self.current_phase = Phase.DRAFT
        self.current_player = PlayerOrder.FIRST
        self.turn = 1

        self.players = [Player(), Player()]

        self._prepare_for_draft()

        return self._build_game_state()

    def step(self, action: Action) -> (GameState, float, bool, dict):
        if self.current_phase == Phase.DRAFT:
            assert type(action) == DraftAction

            current_player = self.players[self.current_player]
            card = current_player.hand[action.chosen_card_index]

            current_player.deck.append(card)
            current_player.hand.clear()

            has_changed_turns = self._next_turn()

            if self.current_phase == Phase.BATTLE:
                self._prepare_for_battle()
        elif self.current_phase == Phase.BATTLE:
            pass

        info = {'turn': self.turn, 'phase': self.current_phase}

        return self._build_game_state(), 0, False, info

    def _next_turn(self) -> bool:
        if self.current_player == PlayerOrder.FIRST:
            self.current_player = PlayerOrder.SECOND

            return False
        else:
            self.current_player = PlayerOrder.FIRST
            self.turn += 1

            if self.turn > self.cards_in_deck:
                self.current_phase = Phase.BATTLE
                self.turn = 1

            return True

    def _prepare_for_draft(self):
        self._draft_cards = self._new_draft()
        self.lanes = (([], []), ([], []))

        current_draft_choices = self._draft_cards[self.turn - 1]

        for player in self.players:
            player.hand.extend(current_draft_choices)

    def _prepare_for_battle(self):
        pass  # todo: implement

    def _build_game_state(self) -> GameState:
        return GameState(self.current_player, self.players, self.lanes)

    def _new_draft(self) -> List[List[Card]]:
        draft = []

        for _ in range(self.cards_in_deck):
            draft.append(np.random.choice(self._cards, 3, replace=False).tolist())

        return draft

    @staticmethod
    def _load_cards() -> List[Card]:
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


class LoCMEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    card_types = {'creature': 0, 'itemGreen': 1, 'itemRed': 2, 'itemBlue': 3}

    def __init__(self, use_draft_history=True, cards_in_deck=30):
        self.state = None
        self.turn = 1

        # self._draft = None

        self.cards_in_state = 33 if use_draft_history else 3
        self.card_features = 16

        self.cards_in_deck = cards_in_deck
        self.state_shape = (self.cards_in_state, self.card_features)

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=self.state_shape,
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(3)

    def reset(self):
        self.turn = 1

        # self.state = self.draft[self.turn - 1]

        return self._convert_state()

    def step(self, action):
        pass

    def render(self, mode='human'):
        print(self._convert_state())

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
