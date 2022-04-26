from abc import ABC, abstractmethod

from gym_locm.engine import State, PlayerOrder


class RewardFunction(ABC):
    @abstractmethod
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        pass


available_rewards = {}


def parse_reward(reward_name: str):
    return available_rewards[reward_name.lower().replace(" ", "-")]
