import gym
import numpy as np

from gym_locm.agents import RandomDraftAgent, RandomBattleAgent
from gym_locm.engine import Phase, State, Action, PlayerOrder, MalformedActionError
from gym_locm.envs.base_env import LOCMEnv
from gym_locm.exceptions import GameIsEndedError


class LOCMFullGameEnv(LOCMEnv):
    def __init__(self, seed=None, k=3, n=30):
        super().__init__(seed=seed, k=k, n=n)

        self.choices = ([], [])

        cards_in_draft_state = self.k
        cards_in_battle_state = 8 + 6 + 6

        player_features = 4  # hp, mana, next_rune, next_draw
        card_features = 16

        self.state_shapes = {
            Phase.DRAFT: (cards_in_draft_state, card_features),
            Phase.BATTLE: (player_features * 2 + cards_in_battle_state * card_features,)
        }

        self.observation_spaces = {
            Phase.DRAFT: gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32,
                                        shape=self.state_shapes[Phase.DRAFT]),
            Phase.BATTLE: gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32,
                                         shape=self.state_shapes[Phase.BATTLE])
        }

        self.action_spaces = {
            Phase.DRAFT: gym.spaces.Discrete(self.k),
            Phase.BATTLE: gym.spaces.Discrete(163)
        }

    @property
    def observation_space(self):
        return self.observation_spaces[self.state.phase]

    @property
    def action_space(self):
        return self.action_spaces[self.state.phase]

    def reset(self):
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # recover random state from current state obj
        random_state = self.state.np_random

        # start a brand new game
        self.state = State(items=self.items)

        # apply random state
        self.state.np_random = random_state

        # empty draft choices
        self.choices = ([], [])

        return self.encode_state()

    def step(self, action):
        """Makes an action in the game."""
        if self.state.phase == Phase.ENDED:
            raise GameIsEndedError()

        # check if an action object or an integer was passed
        if not isinstance(action, Action):
            try:
                action = int(action)
            except ValueError:
                error = f"Action should be an action object " \
                    f"or an integer, not {type(action)}"

                raise MalformedActionError(error)

            action = self.decode_action(action)

        # less property accesses
        state = self.state

        if state.phase == Phase.DRAFT:
            # find chosen card and keep track of it
            chosen_index = action.origin if 0 <= action.origin < self.k else 0
            chosen_card = state.current_player.hand[chosen_index]

            self.choices[state.current_player.id].append(chosen_card)

        # execute the action
        if action is not None:
            state.act(action)

        # init return info
        winner = state.winner
        reward = 0
        done = winner is not None
        info = {'phase': state.phase,
                'turn': state.turn,
                'winner': winner}

        if winner is not None:
            reward = 1 if winner == PlayerOrder.FIRST else -1

            del info['turn']

        return self.encode_state(), reward, done, info

    def _encode_state_battle(self):
        encoded_state = np.full(self.state_shapes[Phase.BATTLE],
                                0, dtype=np.float32)

        p0, p1 = self.state.current_player, self.state.opposing_player

        dummy_card = [0] * 16

        def fill_cards(card_list, up_to):
            remaining_cards = up_to - len(card_list)

            return card_list + [dummy_card for _ in range(remaining_cards)]

        all_cards = []

        locations = p0.hand, p0.lanes[0], p0.lanes[1], p1.lanes[0], p1.lanes[1]
        card_limits = 8, 3, 3, 3, 3

        for location, card_limit in zip(locations, card_limits):
            # convert all cards to features
            location = list(map(self.encode_card, location))

            # add dummy cards up to the card limit
            location = fill_cards(location, up_to=card_limit)

            # add to card list
            all_cards.extend(location)

        # players info
        encoded_state[:8] = self.encode_players(p0, p1)
        encoded_state[8:] = np.array(all_cards).flatten()

        return encoded_state

    def _encode_state_draft(self):
        encoded_state = np.full(self.state_shapes[Phase.DRAFT],
                                0, dtype=np.float32)

        card_choices = self.state.current_player.hand[0:self.k]

        for i in range(len(card_choices)):
            encoded_state[-(self.k - i)] = self.encode_card(card_choices[i])

        return encoded_state


class LOCMFullGameSingleEnv(LOCMFullGameEnv):
    def __init__(self,
                 draft_agent=RandomDraftAgent(),
                 battle_agent=RandomBattleAgent(),
                 play_first=True,
                 seed=None):
        # init the env
        super().__init__(seed=seed)

        # also init the new parameters
        self.play_first = play_first
        self.agents = {
            Phase.DRAFT: draft_agent,
            Phase.BATTLE: battle_agent
        }

    @property
    def agent(self):
        return self.agents[self.state.phase]

    def reset(self):
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset what is needed
        encoded_state = super().reset()

        # also reset the agents
        for agent in self.agents.values():
            agent.reset()

        # if playing second, have first player play
        if not self.play_first:
            while self.state.current_player.id != PlayerOrder.SECOND:
                super().step(self.agent.act(self.state))

        return encoded_state

    def step(self, action):
        """Makes an action in the game."""
        player = self.state.current_player.id

        # do the action
        state, reward, done, info = super().step(action)

        # have opponent play until its player's turn or there's a winner
        while self.state.current_player.id != player and self.state.winner is None:
            state, reward, done, info = super().step(self.agent.act(self.state))

            if info['invalid'] and not done:
                state, reward, done, info = super().step(0)
                break

        if not self.play_first:
            reward = -reward

        return state, reward, done, info
