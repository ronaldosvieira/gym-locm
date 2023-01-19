from abc import ABC, abstractmethod

import numpy as np


class Phase(ABC):
    def __init__(self, rng: np.random.Generator, items=True):
        self.rng = rng
        self.items = items

        self.turn = 1

        self.__available_actions = None
        self.__action_mask = None

    @abstractmethod
    def available_actions(self):
        pass

    @abstractmethod
    def action_mask(self):
        pass

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def new_turn(self):
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def to_native_input(self):
        pass


class DeckBuildingPhase(Phase, ABC):
    pass


class DraftPhase(DeckBuildingPhase):

    def available_actions(self):
        pass

    def action_mask(self):
        pass

    def prepare(self):
        pass

    def act(self):
        pass

    def new_turn(self):
        pass

    def clone(self):
        pass

    def to_native_input(self):
        pass


class ConstructedPhase(DeckBuildingPhase):

    def available_actions(self):
        pass

    def action_mask(self):
        pass

    def prepare(self):
        pass

    def act(self):
        pass

    def new_turn(self):
        pass

    def clone(self):
        pass

    def to_native_input(self):
        pass


class BattlePhase(Phase, ABC):
    pass


class Version12BattlePhase(Phase):

    def available_actions(self):
        pass

    def action_mask(self):
        pass

    def prepare(self):
        pass

    def act(self):
        pass

    def new_turn(self):
        pass

    def clone(self):
        pass

    def to_native_input(self):
        pass
