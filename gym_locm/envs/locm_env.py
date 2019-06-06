from typing import List, Optional

import gym
import copy
import numpy as np

from enum import Enum, IntEnum


instance_counter = -1


def _next_instance_id():
    global instance_counter

    instance_counter += 1

    return instance_counter


class Phase(Enum):
    DRAFT = 0
    BATTLE = 1


class PlayerOrder(IntEnum):
    FIRST = 0
    SECOND = 1


class Lane(IntEnum):
    LEFT = 0
    RIGHT = 1


class BattleActionType(Enum):
    SUMMON = 0
    ATTACK = 1
    USE = 2


class FullHandError(Exception):
    pass


class EmptyDeckError(Exception):
    pass


class NotEnoughManaError(Exception):
    pass


class MalformedActionError(Exception):
    pass


class FullLaneError(Exception):
    pass


class Player:
    def __init__(self):
        self.health = 30
        self.base_mana = 1
        self.bonus_mana = 0
        self.mana = self.base_mana
        self.next_rune = 25
        self.bonus_draw = 0
        self.draw = 1

        self.deck = []
        self.hand = []

    def draw(self, amount=1):
        for _ in range(amount):
            # TODO: check which exception should have precedence

            if len(self.hand) >= 8:
                raise FullHandError()

            if len(self.deck) == 0:
                raise EmptyDeckError()

            self.hand.append(self.deck.pop())

    def damage(self, amount):
        self.health -= amount

        if self.health <= self.next_rune:
            self.next_rune -= 5
            self.bonus_draw += 1


class Card:
    def __init__(self, id, name, type, cost, attack, defense, keywords,
                 player_hp, enemy_hp, card_draw):
        self.id = id
        self.instance_id = None
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

    def make_copy(self):
        card = copy.copy(self)

        card.instance_id = _next_instance_id()

        return card

    def __eq__(self, other):
        return self.instance_id is not None \
               and self.instance_id == other.instance_id


class Creature(Card):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.can_attack = False


class Item(Card):
    pass


class GreenItem(Item):
    pass


class RedItem(Item):
    pass


class BlueItem(Item):
    pass


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
    def __init__(self, type, origin, target):
        self.type = type
        self.origin = origin
        self.target = target


class Game:
    _draft_cards: List[List[Card]]
    current_player: PlayerOrder
    current_phase: Phase

    def __init__(self, cards_in_deck=30):
        self.cards_in_deck = cards_in_deck

        self._cards = self._load_cards()
        self.players = ()
        self.lanes = ()
        self.turn = -1

        self.reset()

    def reset(self) -> GameState:
        self.current_phase = Phase.DRAFT
        self.current_player = PlayerOrder.FIRST
        self.turn = 1

        self.players = (Player(), Player())

        self._prepare_for_draft()

        return self._build_game_state()

    def step(self, action: Action) -> (GameState, bool, dict):
        if self.current_phase == Phase.DRAFT:
            assert type(action) == DraftAction

            self._act_on_draft(action)

            self._next_turn()

            if self.current_phase == Phase.DRAFT:
                self._new_draft_turn()
            elif self.current_phase == Phase.BATTLE:
                self._prepare_for_battle()
                self._new_battle_turn()

        elif self.current_phase == Phase.BATTLE:
            assert type(action) == BattleAction

            self._act_on_battle(action)

            self._next_turn()

            self._new_battle_turn()

        new_state = self._build_game_state()
        has_ended = False
        info = {'turn': self.turn, 'phase': self.current_phase}

        if self.players[PlayerOrder.FIRST].health <= 0:
            info['winner'] = PlayerOrder.SECOND
            has_ended = True
        elif self.players[PlayerOrder.SECOND].health <= 0:
            info['winner'] = PlayerOrder.FIRST
            has_ended = True

        return new_state, has_ended, info

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
        """Prepare all game components for a draft phase"""
        self._draft_cards = self._new_draft()
        self.lanes = (([], []), ([], []))

        current_draft_choices = self._draft_cards[self.turn - 1]

        for player in self.players:
            player.hand = current_draft_choices

    def _prepare_for_battle(self):
        """Prepare all game components for a battle phase"""
        self.lanes = (([], []), ([], []))

        for player in self.players:
            np.random.shuffle(player.deck)
            player.draw(4)
            player.base_mana = 0

        second_player = self.players[PlayerOrder.FIRST]
        second_player.draw()
        second_player.bonus_mana = 1

    def _new_draft_turn(self):
        """Initialize a draft turn"""
        current_draft_choices = self._draft_cards[self.turn - 1]

        for player in self.players:
            player.hand = current_draft_choices

    def _new_battle_turn(self):
        """Initialize a battle turn"""
        current_player = self.players[self.current_player]

        if current_player.base_mana < 12:
            current_player.base_mana += 1

        current_player.mana = current_player.base_mana \
            + current_player.bonus_mana

        amount_to_draw = current_player.draw \
            + current_player.bonus_draw
        current_player.bonus_draw = 0

        for creature in self.lanes[self.current_player][Lane.LEFT]:
            creature.can_attack = True

        for creature in self.lanes[self.current_player][Lane.RIGHT]:
            creature.can_attack = True

        try:
            current_player.draw(amount_to_draw)
        except FullHandError:
            pass
        except EmptyDeckError:
            amount_of_damage = current_player.health \
                               - current_player.next_rune
            current_player.damage(amount_of_damage)

    def _act_on_draft(self, action: DraftAction):
        """Execute the action intended by the player in this draft turn"""
        current_player = self.players[self.current_player]

        card = current_player.hand[action.chosen_card_index]

        current_player.deck.append(card.make_copy())

    def _act_on_battle(self, action: BattleAction):
        """Execute the actions intended by the player in this battle turn"""
        current_player = self.players[self.current_player]

        try:
            if action.origin.cost > current_player.mana:
                raise NotEnoughManaError()

            if action.type == BattleActionType.SUMMON:
                pass  # TODO: implement
            elif action.type == BattleActionType.ATTACK:
                pass  # TODO: implement
            elif action.type == BattleActionType.USE:
                pass  # TODO: implement
            else:
                raise MalformedActionError("Invalid action type.")

            current_player.mana -= action.origin.cost
        except (NotEnoughManaError, MalformedActionError, FullLaneError):
            pass

        if current_player.mana == 0:
            current_player.bonus_mana = 0

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
            type_mapping = {'creature': Creature, 'itemRed': RedItem,
                            'itemGreen': GreenItem, 'itemBlue': BlueItem}

            for card in raw_cards:
                id, name, card_type, cost, attack, defense, \
                keywords, player_hp, enemy_hp, card_draw, _ = \
                    map(str.strip, card.split(';'))

                card_class = type_mapping[card_type]

                cards.append(card_class(int(id), name, card_type, int(cost),
                                        int(attack), int(defense), keywords,
                                        int(player_hp), int(enemy_hp),
                                        int(card_draw)))

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
