from abc import ABC, abstractmethod

from gym_locm.agents import CoacBattleAgent
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


class PlayerHealthRewardFunction(RewardFunction):
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        return state.players[for_player].health / 30


class OpponentHealthRewardFunction(RewardFunction):
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        return -max(0, state.players[for_player.opposing()].health) / 30


class PlayerBoardPresenceRewardFunction(RewardFunction):
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        return sum(creature.attack for lane in state.players[for_player].lanes for creature in lane)


class OpponentBoardPresenceRewardFunction(RewardFunction):
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        return -sum(creature.attack for lane in state.players[for_player.opposing()].lanes for creature in lane)


class CoacRewardFunction(RewardFunction):
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        signal = 1 if state.current_player.id == for_player else -1

        return min(1, max(-1, signal * CoacBattleAgent.eval_state(state) / 2000))


available_rewards = {
    "win-loss": WinLossRewardFunction,
    "player-health": PlayerHealthRewardFunction,
    "opponent-health": OpponentHealthRewardFunction,
    "player-board-presence": PlayerBoardPresenceRewardFunction,
    "opponent-board-presence": OpponentBoardPresenceRewardFunction,
    "coac": CoacRewardFunction
}


def parse_reward(reward_name: str):
    return available_rewards[reward_name.lower().replace(" ", "-")]
