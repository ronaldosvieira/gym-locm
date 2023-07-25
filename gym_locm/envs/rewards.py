from abc import ABC, abstractmethod

from gym_locm.engine import State, PlayerOrder, Creature


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
        return sum(
            creature.attack
            for lane in state.players[for_player].lanes
            for creature in lane
        )


class OpponentBoardPresenceRewardFunction(RewardFunction):
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        return -sum(
            creature.attack
            for lane in state.players[for_player.opposing()].lanes
            for creature in lane
        )


class CoacRewardFunction(RewardFunction):
    @staticmethod
    def _eval_creature(creature) -> int:
        score = 0

        if creature.attack > 0:
            score += 20
            score += creature.attack * 10
            score += creature.defense * 5

            if creature.has_ability("W"):
                score += creature.attack * 5

            if creature.has_ability("L"):
                score += 20

        if creature.has_ability("G"):
            score += 9

        return score

    @staticmethod
    def eval_state(state) -> int:
        score = 0

        player, enemy = state.current_player, state.opposing_player

        for lane in player.lanes:
            for creature in lane:
                score += CoacRewardFunction._eval_creature(creature)

        for lane in enemy.lanes:
            for creature in lane:
                score -= CoacRewardFunction._eval_creature(creature)

        for card in player.hand:
            if not isinstance(card, Creature):
                score += 21  # todo: discover what passed means

        if len(player.hand) + player.bonus_draw + 1 <= 8:
            score += (player.bonus_draw + 1) * 5

        score += player.health * 2
        score -= enemy.health * 2

        if player.health < 5:
            score -= 100

        if enemy.health <= 0:
            score += 100000
        elif player.health <= 0:
            score -= 100000

        return score

    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        signal = 1 if state.current_player.id == for_player else -1

        reward = signal * CoacRewardFunction.eval_state(state) / 2000

        return min(1, max(-1, reward))


available_rewards = {
    "win-loss": WinLossRewardFunction,
    "player-health": PlayerHealthRewardFunction,
    "opponent-health": OpponentHealthRewardFunction,
    "player-board-presence": PlayerBoardPresenceRewardFunction,
    "opponent-board-presence": OpponentBoardPresenceRewardFunction,
    "coac": CoacRewardFunction,
}


def parse_reward(reward_name: str):
    return available_rewards[reward_name.lower().replace(" ", "-")]
