from abc import ABC, abstractmethod

from gym_locm.engine import State, PlayerOrder, Creature


class RewardFunction(ABC):
    @abstractmethod
    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST):
        pass

    def is_action_based(self) -> bool:
        """Return True if this function computes reward from the action taken.

        Action-based reward functions receive ``calculate_action_reward()``
        instead of the usual ``calculate(before) / calculate(after)`` delta
        pattern.  Defaults to False (state-delta behaviour).
        """
        return False

    def calculate_action_reward(
        self, state: State, action_id: int, for_player: PlayerOrder = PlayerOrder.FIRST
    ) -> float:
        """Reward for taking *action_id* in the given pre-action *state*.

        Called by ``LOCMBattleEnv.step()`` **only** when ``is_action_based()``
        returns True, with the state **before** ``state.act()`` is called.
        The implementation is responsible for advancing any per-step internal
        state (e.g. an LSTM hidden state).

        Args:
            state:     Pre-action game state (current player is the acting agent).
            action_id: Integer action index in [0, 144] passed to ``step()``.
                       None when the caller supplied an Action object directly
                       (internal path); implementations should return 0.0 in
                       that case.
            for_player: Player perspective (unused by most implementations).

        Returns:
            Scalar reward ∈ [0, 1] for the chosen action.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declared is_action_based()=True "
            "but did not implement calculate_action_reward()"
        )

    def reset(self) -> None:
        """Reset any per-episode internal state (e.g. LSTM hidden state).

        Called by ``LOCMBattleEnv.reset()`` at the start of every episode.
        The default implementation is a no-op.
        """
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
    def eval_state(state, for_player: PlayerOrder = PlayerOrder.FIRST) -> int:
        score = 0

        player = state.players[for_player]
        enemy = state.players[for_player.opposing()]

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
        reward = CoacRewardFunction.eval_state(state, for_player) / 2000

        return min(1, max(-1, reward))


class ByteRLPolicyRewardFunction(RewardFunction):
    """Rewards the agent proportionally to ByteRL's action probability.

    For each action *a* taken in state *s*, the reward is π_ByteRL(a | s):
    the probability ByteRL would assign to that action given the same game
    position.  This encourages the training agent to match ByteRL's action
    preferences without fully cloning its behaviour.

    ByteRL's internal LSTM state is advanced step-by-step in sync with the
    episode so that temporal context is preserved throughout the game.  Call
    ``reset()`` (done automatically by ``LOCMBattleEnv.reset()``) at the
    start of each episode to reinitialise the LSTM and card-recorder state.

    Example
    -------
    If ByteRL's policy for some state is π(a | s) = [0.10, 0.70, 0.05, 0.15]
    over the four valid actions, then the agent is rewarded with 0.10, 0.70,
    0.05, or 0.15 depending on which action it chose.
    """

    def __init__(self):
        from gym_locm.agents.byterl.main import ensure_weights_are_available
        from gym_locm.agents.byterl.agent import ByterlAgent

        ensure_weights_are_available()
        self._ByterlAgent = ByterlAgent
        self._agent = ByterlAgent()

    # ------------------------------------------------------------------
    # RewardFunction interface
    # ------------------------------------------------------------------

    def is_action_based(self) -> bool:
        return True

    def calculate(self, state: State, for_player: PlayerOrder = PlayerOrder.FIRST) -> float:
        # Not used — rewards come exclusively from calculate_action_reward().
        return 0.0

    def calculate_action_reward(
        self, state: State, action_id: int, for_player: PlayerOrder = PlayerOrder.FIRST
    ) -> float:
        """Return ByteRL's probability for *action_id* in the pre-action state.

        The method advances ByteRL's LSTM state as a side-effect so that
        subsequent calls within the same episode retain temporal context.

        Unlike the competition version of ByteRL which needed a CardRecorder to
        reconstruct deck information, gym-locm exposes the full State (including
        ``state.current_player.deck``), so no preprocessing is necessary —
        ByterlAgent can receive ``state.clone()`` directly.

        Args:
            state:     Pre-action game state.  ``state.current_player`` is
                       the training agent (i.e. the player whose turn it is).
            action_id: Integer action index in [0, 144].  When *None*
                       (caller passed an Action object directly) returns 0.0.
            for_player: Ignored — ByteRL always evaluates from the current
                        player's perspective.

        Returns:
            Probability in [0.0, 1.0] that ByteRL would choose *action_id*.
        """
        from gym_locm.engine import Phase

        if action_id is None or state.phase != Phase.BATTLE:
            return 0.0

        # gym-locm maintains full state, so we can query ByteRL directly.
        # Clone to avoid mutating the live game state.
        probs = self._agent.get_policy_probs(state.clone())
        return float(probs[action_id])

    def reset(self) -> None:
        """Reinitialise ByteRL's LSTM state for a new episode."""
        self._agent = self._ByterlAgent()


available_rewards = {
    "win-loss": WinLossRewardFunction,
    "player-health": PlayerHealthRewardFunction,
    "opponent-health": OpponentHealthRewardFunction,
    "player-board-presence": PlayerBoardPresenceRewardFunction,
    "opponent-board-presence": OpponentBoardPresenceRewardFunction,
    "coac": CoacRewardFunction,
    "byterl-policy": ByteRLPolicyRewardFunction,
}


def parse_reward(reward_name: str):
    return available_rewards[reward_name.lower().replace(" ", "-")]
