import copy
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from gym_locm.engine import (
    Card,
    Creature,
    Action,
    ActionType,
    PlayerOrder,
    get_locm12_card_list,
)
from gym_locm.util import is_it


class Phase(ABC):
    def __init__(self, state, rng: np.random.Generator, items=True):
        self.state = state
        self.rng = rng
        self.items = items

        self.turn = 1
        self.ended = False

        self._current_player = None
        self.__available_actions = None
        self.__action_mask = None

    @abstractmethod
    def available_actions(self) -> Tuple[Action]:
        pass

    @abstractmethod
    def action_mask(self) -> Tuple[bool]:
        pass

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def act(self, action: Action):
        pass

    @abstractmethod
    def _next_turn(self):
        pass

    def clone(self):
        return copy.deepcopy(self)


class DeckBuildingPhase(Phase, ABC):
    def __init__(self, state, rng, items=True):
        super().__init__(state, rng, items)

        # get references of the players' deck
        self.decks = state.players[0].deck, state.players[1].deck


class DraftPhase(DeckBuildingPhase):
    def __init__(self, state, rng, k=3, n=30, items=True):
        super().__init__(state, rng, items)

        self.k, self.n = k, n

        self._draft_cards = None

        self.__available_actions = self.__available_actions = tuple(
            [Action(ActionType.PICK, i) for i in range(self.k)]
        )
        self.__action_mask = tuple([True] * self.k)

    def available_actions(self) -> Tuple[Action]:
        return self.__available_actions if not self.ended else ()

    def action_mask(self) -> Tuple[bool]:
        return self.__action_mask if not self.ended else ()

    def prepare(self):
        # initialize current player pointer
        self._current_player = PlayerOrder.FIRST

        # initialize random draft cards
        self._draft_cards = self._new_draft()

    def _new_draft(self):
        # retrieve card list
        cards = list(get_locm12_card_list())

        # if items are not wanted, filter them out of the card list
        if not self.items:
            cards = list(filter(is_it(Creature), cards))

        # get 60 random cards from the card list
        self.rng.shuffle(cards)
        pool = cards[:60]

        # get 3 random cards without replacement for each turn
        draft = []

        for _ in range(self.n):
            self.rng.shuffle(pool)

            draft.append(pool[: self.k])

        return draft

    def act(self, action: Action):
        """Execute the action intended by the player in this draft turn"""
        # get chosen card
        chosen_index = action.origin if action.origin is not None else 0
        card = self.current_choices[chosen_index]

        # add chosen card to player's deck
        self.decks[self._current_player].append(card)

        # trigger next turn
        self._next_turn()

    def _next_turn(self):
        # handle turn change
        if self._current_player == PlayerOrder.FIRST:
            self._current_player = PlayerOrder.SECOND
        else:
            if self.turn < self.n:
                self._current_player = PlayerOrder.FIRST

                self.turn += 1

                for player in self.state.players:
                    player.hand = self.current_choices
            else:
                self._current_player = None
                self.ended = True

    @property
    def current_choices(self) -> List[Card]:
        try:
            return self._draft_cards[self.turn - 1]
        except IndexError:
            return []


class ConstructedPhase(DeckBuildingPhase):
    def available_actions(self) -> Tuple[Action]:
        pass

    def action_mask(self) -> Tuple[bool]:
        pass

    def prepare(self):
        pass

    def act(self, action: Action):
        pass

    def _next_turn(self):
        pass


class BattlePhase(Phase, ABC):
    pass


class Version12BattlePhase(Phase):
    def available_actions(self) -> Tuple[Action]:
        pass

    def action_mask(self) -> Tuple[bool]:
        pass

    def prepare(self):
        pass

    def act(self, action: Action):
        pass

    def _next_turn(self):
        pass
