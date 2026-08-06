from abc import ABC, abstractmethod
import os

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


DISTILLED_VF_URL = "https://raw.githubusercontent.com/ronaldosvieira/byterl-vf-distillation/refs/heads/master/byterl_vf.npz"

class ByteRLValueRewardFunction(RewardFunction):
    """
    Reward function that uses a distilled NumPy MLP to approximate the value
    of a given state according to the ByteRL agent. 
    """
    def __init__(self):
        super().__init__()
        self.weights = None
        self.env = None

    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        if getattr(state, 'version', "1.2") != "1.5":
            raise ValueError("ByteRLValueRewardFunction is only supported for LOCM 1.5.")
            
        if self.weights is None:
            import numpy as np
            import os
            import urllib.request
            
            model_path = os.path.join(
                os.path.dirname(__file__),
                "byterl_vf.npz"
            )
            
            if not os.path.exists(model_path):
                print(f"Distilled ByteRL VF weights not found at {model_path}.")
                print(f"Downloading weights from {DISTILLED_VF_URL}...")
                urllib.request.urlretrieve(DISTILLED_VF_URL, model_path)
                print("Download complete.")
                
            self.weights = np.load(model_path)
            
            # We need an encoder to get the flat state. We can use a dummy env.
            from gym_locm.envs.battle import LOCMBattleEnv
            self.env = LOCMBattleEnv(version="1.5")
            
        # We need to construct the 265-dim observation from the state
        # The true state has to be encoded by the battle env encoder
        self.env.state = state.clone()
        obs = self.env.encode_state()
        
        import numpy as np
        
        def relu(x):
            return np.maximum(0, x)
            
        x = obs
        
        num_layers = int(self.weights['num_layers'][0])
        
        # Loop through hidden layers
        for i in range(num_layers):
            x = np.dot(x, self.weights[f'w{i+1}'].T) + self.weights[f'b{i+1}']
            x = relu(x)
            
        # Final output layer
        x = np.dot(x, self.weights[f'w{num_layers+1}'].T) + self.weights[f'b{num_layers+1}']
        
        value = float(x[0]) if isinstance(x, np.ndarray) and x.size > 0 else float(x)
            
        # If calculating for the opposing player, negate the value
        # Note: ByteRL value is trained from the perspective of the current player.
        # So we should be careful here. For potential based shaping: F(s, s') = V(s') - V(s)
        # We just return the state's value for the given player
        if state.current_player.id == for_player:
            return value
        else:
            return -value


available_rewards = {
    "win-loss": WinLossRewardFunction,
    "player-health": PlayerHealthRewardFunction,
    "opponent-health": OpponentHealthRewardFunction,
    "player-board-presence": PlayerBoardPresenceRewardFunction,
    "opponent-board-presence": OpponentBoardPresenceRewardFunction,
    "coac": CoacRewardFunction,
    "byterl-vf": ByteRLValueRewardFunction,
}


def parse_reward(reward_name: str):
    return available_rewards[reward_name.lower().replace(" ", "-")]
