import gymnasium as gym
import numpy as np

from gym_locm.agents import RandomDraftAgent, RandomBattleAgent
from gym_locm.engine import Phase, Action, PlayerOrder, ActionType
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
        use_average_deck=True,
        dict_observations=False,
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
        self.dict_observations = dict_observations

        player_features = 3 if self.version == "1.5" else 4
        cards_in_hand = 8
        deck_size_features = 1
        hand_size_features = 1
        card_features = 17 if self.items else 13
        friendly_cards_on_board = 6
        friendly_board_card_features = 9 if not self.dict_observations else 17
        enemy_cards_on_board = 6
        enemy_board_card_features = 8 if not self.dict_observations else 17

        player_features += 1 if version == "1.2" else 0
        card_features -= 1 if version == "1.2" else 0

        self.state_shape = (
            player_features * 2  # player and opponent's stats
            
            + deck_size_features * 2  # player and opponent's deck size
            + card_features * int(self.use_average_deck)  # player's deck content
            
            + hand_size_features * 2  # player and opponent's hand size
            + cards_in_hand * card_features  # player's hand content
            
            + friendly_cards_on_board * friendly_board_card_features  # player's battlefield content
            + enemy_cards_on_board * enemy_board_card_features  # opponent's battlefield content
        )
        
        if dict_observations:
            self.observation_space = gym.spaces.Dict({
                "player_stats": gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
                "player_deck": gym.spaces.Box(low=-1.0, high=1.0, shape=(30, card_features,), dtype=np.float32),
                "player_hand": gym.spaces.Box(low=-1.0, high=1.0, shape=(cards_in_hand, card_features,), dtype=np.float32),
                "player_lane0": gym.spaces.Box(low=-1.0, high=1.0, shape=(3, friendly_board_card_features,), dtype=np.float32),
                "player_lane1": gym.spaces.Box(low=-1.0, high=1.0, shape=(3, friendly_board_card_features,), dtype=np.float32),
                
                "opponent_stats": gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
                "opponent_lane0": gym.spaces.Box(low=-1.0, high=1.0, shape=(3, enemy_board_card_features,), dtype=np.float32),
                "opponent_lane1": gym.spaces.Box(low=-1.0, high=1.0, shape=(3, enemy_board_card_features,), dtype=np.float32),
                
                "action_mask": gym.spaces.MultiBinary(145 if self.items else 41),
            })
        else:
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

        self._update_player_decks()

        self._encoded_state = np.zeros(self.state_shape, dtype=np.float32)

    def _update_player_decks(self):
        self.player_decks = [None, None]
        self.player_decks_mean = [None, None]

        for player in self.state.players:
            self.player_decks[player.id] = list(player.deck + player.hand)
            assert len(self.player_decks[player.id]) == 30

            # Precalculate average deck features
            card_features = 17 if self.items else 13
            if self.version == "1.2":
                card_features -= 1

            deck_cards = self.player_decks[player.id]
            encoded_deck = []
            for card in deck_cards:
                features = self.encode_card(card, version=self.version)
                if not self.items:
                    features = features[4:]
                encoded_deck.append(features)

            self.player_decks_mean[player.id] = np.mean(encoded_deck, axis=0)

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

        reward_before = [
            weight * function.calculate(state, for_player=PlayerOrder.FIRST)
            for function, weight in zip(self.reward_functions, self.reward_weights)
        ]

        # pre-action metrics capture
        is_pass = False
        is_attack = False
        unspent_mana, hand_size, lane0_val, lane1_val = 0, 0, 0, 0
        skipped_dominant_action = False

        if action is not None:
            if action.type == ActionType.PASS:
                is_pass = True
                cp = state.current_player
                unspent_mana = cp.mana
                hand_size = len(cp.hand)
                lane0_val = sum(c.attack + c.defense for c in cp.lanes[0])
                lane1_val = sum(c.attack + c.defense for c in cp.lanes[1])
                
                # Check for skipped dominant actions
                for a in state.available_actions:
                    if a.type == ActionType.ATTACK:
                        if a.target is None:  # Face attack available
                            skipped_dominant_action = True
                            break
                        # Check if attacking 0-attack creature
                        op = state.opposing_player
                        target_card = None
                        for lane in op.lanes:
                            for c in lane:
                                if c.instance_id == a.target:
                                    target_card = c
                                    break
                            if target_card is not None:
                                break
                        if target_card is not None and target_card.attack == 0:
                            skipped_dominant_action = True
                            break
            elif action.type == ActionType.ATTACK:
                is_attack = True

        # execute the action
        if action is not None:
            state.act(action)
        else:
            state.was_last_action_invalid = True

        reward_after = [
            weight * function.calculate(state, for_player=PlayerOrder.FIRST)
            for function, weight in zip(self.reward_functions, self.reward_weights)
        ]

        # build return info
        winner = state.winner

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
            "health_diff": (
                state.players[0].health - state.players[1].health,
                state.players[1].health - state.players[0].health,
            )
        }

        # inject new metrics into info
        if action is not None and not state.was_last_action_invalid:
            if is_pass:
                info["turn_mana"] = unspent_mana
                info["turn_hand_size"] = hand_size
                info["lane0_value"] = lane0_val
                info["lane1_value"] = lane1_val
                if skipped_dominant_action:
                    info["skipped_dominant_action"] = True
            elif is_attack:
                if action.resolved_target is None:
                    info["face_attack"] = True
                else:
                    info["creature_attack"] = True
                    attacker = action.resolved_origin
                    defender = action.resolved_target
                    if getattr(defender, 'is_dead', False) and not getattr(attacker, 'is_dead', False):
                        info["favorable_trade"] = True

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
        self._update_player_decks()

        self.rewards.append(0.0)

        return self.encode_state(), {}

    def _encode_state_deck_building(self):
        pass

    def _encode_state_battle(self):
        if self.dict_observations:
            return self._encode_state_battle_dict_obs()
        else:
            return self._encode_state_battle_box_obs()

    def _encode_state_battle_dict_obs(self):
        p0, p1 = self.state.current_player, self.state.opposing_player
        
        def fill_cards(card_list, up_to, features):
            remaining_cards = up_to - len(card_list)

            return card_list + [[0] * features for _ in range(remaining_cards)]
        
        card_features = 17 if self.items else 13

        if self.version == "1.2":
            card_features -= 1
        
        players_info = self.encode_players(
            p0, p1, version=self.version
        )
        
        player_hand = list(map(lambda c: self.encode_card(c, version=self.version), p0.hand))
        player_deck = list(map(lambda c: self.encode_card(c, version=self.version), p0.deck))
        
        if not self.items:
            player_hand = list(map(lambda c: c[4:], player_hand))
            player_deck = list(map(lambda c: c[4:], player_deck))

        player_hand = fill_cards(player_hand, up_to=8, features=card_features)
        player_deck = fill_cards(player_deck, up_to=30, features=card_features)
        
        encode_card = lambda c: self.encode_card(c, version=self.version)

        # encode_enemy_card_on_board is used on purpose to not include can_attack information
        player_lane0 = list(map(encode_card, p0.lanes[0]))
        player_lane0 = fill_cards(player_lane0, up_to=3, features=17)
        
        player_lane1 = list(map(encode_card, p0.lanes[1]))
        player_lane1 = fill_cards(player_lane1, up_to=3, features=17)

        opponent_lane0 = list(map(encode_card, p1.lanes[0]))
        opponent_lane0 = fill_cards(opponent_lane0, up_to=3, features=17)
        
        opponent_lane1 = list(map(encode_card, p1.lanes[1]))
        opponent_lane1 = fill_cards(opponent_lane1, up_to=3, features=17)
        
        encoded_state = {
            "player_stats": list(players_info[:3]) + [len(p0.hand) / 8, len(p0.deck) / 30],
            "player_deck": player_deck,
            "player_hand": player_hand,
            "player_lane0": player_lane0,
            "player_lane1": player_lane1,
            
            "opponent_stats": list(players_info[3:]) + [len(p1.hand) / 8, len(p1.deck) / 30],
            "opponent_lane0": opponent_lane0,
            "opponent_lane1": opponent_lane1,

            "action_mask": np.array(list(map(int, self.state.action_mask))),
        }
        
        return encoded_state

    def _encode_state_battle_box_obs(self):
        encoded_state = self._encoded_state.copy()

        p0, p1 = self.state.current_player, self.state.opposing_player

        # players info
        player_features = 6 if self.version == "1.5" else 8

        players_info = self.encode_players(
            p0, p1, version=self.version
        )
        
        encoded_state[:player_features] = players_info

        anchor = player_features

        encoded_state[anchor] = len(p0.deck) / 25
        encoded_state[anchor + 1] = len(p1.deck) / 25
        
        anchor += 2

        encoded_state[anchor] = len(p0.hand) / 8
        encoded_state[anchor + 1] = len(p1.hand) / 8

        anchor += 2

        card_features = 17 if self.items else 13
        if self.version == "1.2":
            card_features -= 1

        offset = anchor
        # convert all cards in hand to features
        for i, card in enumerate(p0.hand):
            if i >= 8:
                break
            features = self.encode_card(card, version=self.version)
            if not self.items:
                features = features[4:]
            encoded_state[offset:offset + card_features] = features
            offset += card_features
        
        # zero out remaining cards in hand
        encoded_state[offset:anchor + 8 * card_features] = 0.0
        offset = anchor + 8 * card_features

        # in current player's lanes
        for lane_id in (0, 1):
            start_lane_offset = offset
            for i, creature in enumerate(p0.lanes[lane_id]):
                if i >= 3:
                    break
                features = self.encode_friendly_card_on_board(creature)
                encoded_state[offset:offset + 9] = features
                offset += 9
            encoded_state[offset:start_lane_offset + 27] = 0.0
            offset = start_lane_offset + 27

        # in opposing player's lanes
        for lane_id in (0, 1):
            start_lane_offset = offset
            for i, creature in enumerate(p1.lanes[lane_id]):
                if i >= 3:
                    break
                features = self.encode_enemy_card_on_board(creature)
                encoded_state[offset:offset + 8] = features
                offset += 8
            encoded_state[offset:start_lane_offset + 24] = 0.0
            offset = start_lane_offset + 24

        if self.use_average_deck:
            encoded_state[-card_features:] = self.player_decks_mean[p0.id]

        return encoded_state

    def get_episode_rewards(self):
        return self.rewards


class LOCMBattleSingleEnv(LOCMBattleEnv):
    def __init__(
        self,
        battle_agent=RandomBattleAgent(),
        deck_building_agents=(RandomDraftAgent(), RandomDraftAgent()),
        play_first=True,
        alternate_roles=False,
        **kwargs,
    ):
        # init the env
        super().__init__(**kwargs)
        
        # manage the deck-building agents before super.__init__ uses them to simulate deck-building
        self._play_first = True
        self.deck_building_agents = deck_building_agents
        self.play_first = play_first

        # also init the battle agent and the new parameters
        self.battle_agent = battle_agent
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
                        # opponent is repeating the same invalid action, raise exception to avoid infinite loop
                        raise Exception(f"Opponent is repeating the same invalid action: {action}.")

                last_opponent_action = action

        self.rewards_single_player.append(0.0)

        return self.encode_state(), info

    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        """Makes an action in the game."""
        player = self.state.current_player.id

        # do the action
        state, reward, terminated, truncated, info = super().step(action)
        total_reward = reward

        was_invalid = info["invalid"]

        last_opponent_action = None

        # have opponent play until its player's turn or there's a winner
        while self.state.current_player.id != player and self.state.winner is None:
            action = self.battle_agent.act(self.state)

            try:
                state, step_reward, terminated, truncated, info = super().step(action)
                total_reward += step_reward
            except ActionError:
                if action == last_opponent_action:
                    # opponent is repeating the same invalid action, raise exception to avoid infinite loop
                    raise Exception(f"Opponent is repeating the same invalid action: {action}.")

            last_opponent_action = action

        info["invalid"] = was_invalid
        reward = total_reward

        if not self.play_first:
            reward = -reward

        try:
            self.rewards_single_player[-1] += reward
        except IndexError:
            self.rewards_single_player = [reward]

        return state, reward, terminated, truncated, info

    def get_episode_rewards(self):
        return self.rewards_single_player

    @property
    def play_first(self) -> bool:
        return self._play_first

    @play_first.setter
    def play_first(self, value: bool):
        if value != self._play_first:
            self._play_first = value
            self.deck_building_agents = (
                self.deck_building_agents[1],
                self.deck_building_agents[0],
            )


class LOCMBattleSelfPlayEnv(LOCMBattleEnv):
    def __init__(
        self,
        play_first=True,
        deck_building_agents=(RandomDraftAgent(), RandomDraftAgent()),
        alternate_roles=True,
        adversary_policy=None,
        **kwargs
    ):
        # init the env
        super().__init__(**kwargs)
        
        # manage the deck-building agents before super.__init__ uses them to simulate deck-building
        self._play_first = True
        self.deck_building_agents = deck_building_agents
        self.play_first = play_first

        # also init the new parameters
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

        last_opponent_action = None

        # if playing second, have first player play
        if not self.play_first:
            while self.state.current_player.id != PlayerOrder.SECOND and self.state.winner is None:
                state = self.encode_state()
                action = self.adversary_policy(state)

                try:
                    state, reward, terminated, truncated, info = super().step(action)
                except ActionError:
                    if action == last_opponent_action:
                        # opponent is repeating the same invalid action, pass the turn instead
                        state, reward, terminated, truncated, info = super().step(0)

                last_opponent_action = action

        self.rewards_single_player.append(0.0)

        return self.encode_state(), info

    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        """Makes an action in the game."""
        player = self.state.current_player.id

        # do the action
        state, reward, terminated, truncated, info = super().step(action)
        total_reward = reward

        was_invalid = info["invalid"]

        # have opponent play until its player's turn or there's a winner
        last_opponent_action = None

        while self.state.current_player.id != player and self.state.winner is None:
            state = self.encode_state()
            action = self.adversary_policy(state)

            try:
                state, step_reward, terminated, truncated, info = super().step(action)
                total_reward += step_reward
            except ActionError:
                if action == last_opponent_action:
                    # opponent is repeating the same invalid action, pass the turn instead
                    state, step_reward, terminated, truncated, info = super().step(0)
                    total_reward += step_reward

            last_opponent_action = action

        info["invalid"] = was_invalid
        reward = total_reward

        if not self.play_first:
            reward = -reward

        try:
            self.rewards_single_player[-1] += reward
        except IndexError:
            self.rewards_single_player = [reward]

        return state, reward, terminated, truncated, info

    def get_episode_rewards(self):
        return self.rewards_single_player

    @property
    def play_first(self) -> bool:
        return self._play_first

    @play_first.setter
    def play_first(self, value: bool):
        if value != self._play_first:
            self._play_first = value
            self.deck_building_agents = (
                self.deck_building_agents[1],
                self.deck_building_agents[0],
            )
