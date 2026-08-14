import random
from abc import ABC, abstractmethod
from gym_locm.engine import Action, ActionType


class Agent(ABC):
    @abstractmethod
    def seed(self, seed):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    def __repr__(self) -> str:
        return type(self).__name__


class PassAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

    def act(self, state):
        return Action(ActionType.PASS)


class RandomAgent(Agent):
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def seed(self, seed):
        self.random.seed(seed)

    def reset(self):
        pass

    def act(self, state):
        index = int(len(state.available_actions) * self.random.random())

        return state.available_actions[index]
    

PassDraftAgent = PassAgent
PassConstructedAgent = PassAgent
PassBattleAgent = PassAgent

RandomDraftAgent = RandomAgent
RandomConstructedAgent = RandomAgent
RandomBattleAgent = RandomAgent