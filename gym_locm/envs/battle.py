import gymnasium as gym
import numpy as np

from gym_locm.agents import RandomDraftAgent, RandomBattleAgent
from gym_locm.engine import Phase, Action, PlayerOrder
from gym_locm.envs.base_env import LOCMEnv
from gym_locm.exceptions import GameIsEndedError, MalformedActionError, ActionError


class LOCMBattleEnv(LOCMEnv):
    metadata = {"render.modes": ["text", "native"]}

    def __init__(
        self,
        deck_building_agents=(RandomDraftAgent(), RandomDraftAgent()),
        return_action_mask=False,
        seed=None,
        items=True,
        k=None,
        n=30,
        reward_functions=("win-loss",),
        reward_weights=(1.0,),
        version="1.5",
        use_average_deck=False,
        render_mode=None,
    ):
        super().__init__(
            seed=seed,
            version=version,
            items=items,
            k=k if k is not None else (120 if version == "1.5" else 3),
            n=n,
            reward_functions=reward_functions,
            reward_weights=reward_weights,
            render_mode=render_mode,
        )

        self.rewards = [0.0]

        self.version = version
        self.deck_building_agents = deck_building_agents

        for agent in self.deck_building_agents:
            agent.reset()
            agent.seed(seed)

        self.return_action_mask = return_action_mask
        self.use_average_deck = use_average_deck

        player_features = 3
        cards_in_hand = 8
        card_features = 17 if self.items else 13
        friendly_cards_on_board = 6
        friendly_board_card_features = 9
        enemy_cards_on_board = 6
        enemy_board_card_features = 8

        player_features += 1 if version == "1.2" else 0
        card_features -= 1 if version == "1.2" else 0

        self.state_shape = (
            player_features * 2
            + cards_in_hand * card_features
            + friendly_cards_on_board * friendly_board_card_features
            + enemy_cards_on_board * enemy_board_card_features
            + card_features * int(self.use_average_deck)
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_shape,), dtype=np.float32
        )

        if self.items:
            # 145 possible actions
            self.action_space = gym.spaces.Discrete(145)
        else:
            # 41 possible actions
            self.action_space = gym.spaces.Discrete(41)

        self._play_through_deck_building_phase()

        self.player_decks = [None, None]

        for player in self.state.players:
            self.player_decks[player.id] = list(player.deck + player.hand)
            assert len(self.player_decks[player.id]) == 30

    def _play_through_deck_building_phase(self):
        while self.state.phase == Phase.DECK_BUILDING:
            if self.version == "1.5":
                agent = self.deck_building_agents[self.state.current_player.id]
                action = agent.act(self.state)

                self.state.act(action)
            else:
                for agent in self.deck_building_agents:
                    action = agent.act(self.state)

                    self.state.act(action)

    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        """Makes an action in the game."""
        # if the battle is finished, there should be no more actions
        if self._battle_is_finished:
            raise GameIsEndedError()

        # check if an action object or an integer was passed
        if not isinstance(action, Action):
            try:
                action = int(action)
            except ValueError:
                error = (
                    f"Action should be an action object "
                    f"or an integer, not {type(action)}"
                )

                raise MalformedActionError(error)

            action = self.decode_action(action)

        # less property accesses
        state = self.state

        self.last_player_rewards[state.current_player.id] = [
            weight * function.calculate(state, for_player=PlayerOrder.FIRST)
            for function, weight in zip(self.reward_functions, self.reward_weights)
        ]

        # execute the action
        if action is not None:
            state.act(action)
        else:
            state.was_last_action_invalid = True

        reward_before = self.last_player_rewards[state.current_player.id]
        reward_after = [
            weight * function.calculate(state, for_player=PlayerOrder.FIRST)
            for function, weight in zip(self.reward_functions, self.reward_weights)
        ]

        # build return info
        winner = state.winner

        if reward_before is None:
            raw_rewards = (0.0,) * len(self.reward_functions)
        else:
            raw_rewards = tuple(
                [after - before for before, after in zip(reward_before, reward_after)]
            )

        reward = sum(raw_rewards)
        terminated = winner is not None
        info = {
            "phase": state.phase,
            "turn": state.turn,
            "winner": winner,
            "invalid": state.was_last_action_invalid,
            "raw_rewards": raw_rewards,
        }

        if self.return_action_mask:
            info["action_mask"] = self.state.action_mask

        self.rewards[-1] += reward

        return self.encode_state(), reward, terminated, False, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.array, dict]:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset the state
        super().reset()

        # reset all agents' internal state
        for agent in self.deck_building_agents:
            agent.reset()
            agent.seed(self._seed)

        self._play_through_deck_building_phase()

        self.rewards.append(0.0)

        return self.encode_state(), {}

    def _encode_state_deck_building(self):
        pass

    def _encode_state_battle(self):
        encoded_state = np.full(self.state_shape, 0, dtype=np.float32)

        p0, p1 = self.state.current_player, self.state.opposing_player

        def fill_cards(card_list, up_to, features):
            remaining_cards = up_to - len(card_list)

            return card_list + [[0] * features for _ in range(remaining_cards)]

        all_cards = []

        # convert all cards in hand to features
        hand = list(map(lambda c: self.encode_card(c, version=self.version), p0.hand))

        # if not using items, clip card type features
        if not self.items:
            hand = list(map(lambda c: c[4:], hand))

        # add dummy cards up to the card limit
        card_features = 17 if self.items else 13

        if self.version == "1.2":
            card_features -= 1

        hand = fill_cards(hand, up_to=8, features=card_features)

        # add to card list
        all_cards.extend([feature for card in hand for feature in card])

        # in current player's lanes
        for location in (p0.lanes[0], p0.lanes[1]):
            # convert all cards to features
            location = list(map(self.encode_friendly_card_on_board, location))

            # add dummy cards up to the card limit
            location = fill_cards(location, up_to=3, features=9)

            # add to card list
            all_cards.extend([feature for card in location for feature in card])

        # in opposing player's lanes
        for location in (p1.lanes[0], p1.lanes[1]):
            # convert all cards to features
            location = list(map(self.encode_enemy_card_on_board, location))

            # add dummy cards up to the card limit
            location = fill_cards(location, up_to=3, features=8)

            # add to card list
            all_cards.extend([feature for card in location for feature in card])

        # players info
        player_features = 6 if self.version == "1.5" else 8

        encoded_state[:player_features] = self.encode_players(
            p0, p1, version=self.version
        )
        if self.use_average_deck:
            encoded_state[player_features:-card_features] = np.array(
                all_cards
            ).flatten()
            encoded_state[-card_features:] = np.array(
                list(
                    map(
                        lambda c: self.encode_card(c, version=self.version),
                        self.player_decks[p0.id],
                    )
                )
            ).mean(axis=0)
        else:
            encoded_state[player_features:] = np.array(all_cards).flatten()

        return encoded_state

    def get_episode_rewards(self):
        return self.rewards


class LOCMBattleSingleEnv(LOCMBattleEnv):
    def __init__(
        self,
        battle_agent=RandomBattleAgent(),
        play_first=True,
        alternate_roles=False,
        **kwargs,
    ):
        # init the env
        super().__init__(**kwargs)

        # also init the battle agent and the new parameters
        self.battle_agent = battle_agent
        self.play_first = play_first
        self.alternate_roles = alternate_roles
        self.rewards_single_player = []

        # reset the battle agent
        # if it was not already reset as a deck-building agent
        if self.battle_agent not in self.deck_building_agents:
            self.battle_agent.reset()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.array, dict]:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        if self.alternate_roles:
            self.play_first = not self.play_first
            self.deck_building_agents = (
                self.deck_building_agents[1],
                self.deck_building_agents[0],
            )

        # reset what is needed
        encoded_state, info = super().reset()

        # also reset the battle agent
        # if it was not already reset as a deck-building agent
        if self.battle_agent not in self.deck_building_agents:
            self.battle_agent.reset()

        # if playing second, have first player play
        last_opponent_action = None

        if not self.play_first:
            while self.state.current_player.id != PlayerOrder.SECOND:
                action = self.battle_agent.act(self.state)

                try:
                    super().step(action)
                except ActionError:
                    if action == last_opponent_action:
                        # opponent is repeating the same invalid action, pass the turn instead
                        super().step(0)

                last_opponent_action = action

        self.rewards_single_player.append(0.0)

        return encoded_state, info

    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        """Makes an action in the game."""
        player = self.state.current_player.id

        # do the action
        state, reward, terminated, truncated, info = super().step(action)

        was_invalid = info["invalid"]

        last_opponent_action = None

        # have opponent play until its player's turn or there's a winner
        while self.state.current_player.id != player and self.state.winner is None:
            action = self.battle_agent.act(self.state)

            try:
                state, reward, terminated, truncated, info = super().step(action)
            except ActionError:
                if action == last_opponent_action:
                    # opponent is repeating the same invalid action, pass the turn instead
                    state, reward, terminated, truncated, info = super().step(0)

            last_opponent_action = action

        info["invalid"] = was_invalid

        if not self.play_first:
            reward = -reward

        try:
            self.rewards_single_player[-1] += reward
        except IndexError:
            self.rewards_single_player = [reward]

        return state, reward, terminated, truncated, info

    def get_episode_rewards(self):
        return self.rewards_single_player


class LOCMBattleSelfPlayEnv(LOCMBattleEnv):
    def __init__(
        self, play_first=True, alternate_roles=True, adversary_policy=None, **kwargs
    ):
        # init the env
        super().__init__(**kwargs)

        # also init the new parameters
        self.play_first = play_first
        self.adversary_policy = adversary_policy
        self.alternate_roles = alternate_roles
        self.rewards_single_player = []

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.array, dict]:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset what is needed
        encoded_state, info = super().reset()

        if self.alternate_roles:
            self.play_first = not self.play_first
            self.deck_building_agents = (
                self.deck_building_agents[1],
                self.deck_building_agents[0],
            )

        last_opponent_action = None

        # if playing second, have first player play
        if not self.play_first:
            while self.state.current_player.id != PlayerOrder.SECOND:
                state = self.encode_state()
                action = self.adversary_policy(state, self.action_mask)

                try:
                    state, reward, terminated, truncated, info = super().step(action)
                except ActionError:
                    if action == last_opponent_action:
                        # opponent is repeating the same invalid action, pass the turn instead
                        state, reward, terminated, truncated, info = super().step(0)

                last_opponent_action = action

        self.rewards_single_player.append(0.0)

        return encoded_state, info

    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        """Makes an action in the game."""
        player = self.state.current_player.id

        # do the action
        state, reward, terminated, truncated, info = super().step(action)

        was_invalid = info["invalid"]

        # have opponent play until its player's turn or there's a winner
        last_opponent_action = None

        while self.state.current_player.id != player and self.state.winner is None:
            state = self.encode_state()
            action = self.adversary_policy(state, self.action_mask)

            try:
                state, reward, terminated, truncated, info = super().step(action)
            except ActionError:
                if action == last_opponent_action:
                    # opponent is repeating the same invalid action, pass the turn instead
                    state, reward, terminated, truncated, info = super().step(0)

            last_opponent_action = action

        info["invalid"] = was_invalid

        if not self.play_first:
            reward = -reward

        try:
            self.rewards_single_player[-1] += reward
        except IndexError:
            self.rewards_single_player = [reward]

        return state, reward, terminated, truncated, info

    def get_episode_rewards(self):
        return self.rewards_single_player
