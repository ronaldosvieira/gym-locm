from abc import ABC, abstractmethod

from gym_locm.engine import State, PlayerOrder


class RewardFunction(ABC):
    @abstractmethod
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        pass


class WinLossRewardFunction(RewardFunction):
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        if state.winner == for_player:
            return 1
        elif state.winner == for_player.opposing():
            return -1
        else:
            return 0


available_rewards = {
    "win-loss": WinLossRewardFunction
}


def parse_reward(reward_name: str):
    return available_rewards[reward_name.lower().replace(" ", "-")]
