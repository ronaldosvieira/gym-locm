from typing import Union

import gym

from gym_locm.agents import *
from gym_locm.engine import *
from gym_locm.envs.base_env import LOCMEnv


class LOCMDraftEnv(LOCMEnv):
    metadata = {'render.modes': ['text', 'native']}

    def __init__(self,
                 battle_agents=(RandomBattleAgent(), RandomBattleAgent()),
                 use_draft_history=False,
                 sort_cards=False,
                 evaluation_battles=1,
                 seed=None):
        super().__init__(seed=seed)

        # init bookkeeping structures
        self.results = []
        self.choices = ([], [])
        self.draft_ordering = list(range(3))

        self.battle_agents = battle_agents

        for battle_agent in self.battle_agents:
            battle_agent.reset()
            battle_agent.seed(seed)

        self.evaluation_battles = evaluation_battles
        self.sort_cards = sort_cards
        self.use_draft_history = use_draft_history

        self.cards_in_state = 33 if use_draft_history else 3
        self.card_features = 16

        # (30 cards already chosen + 3 current choices) x (16 card features)
        self.state_shape = (self.cards_in_state, self.card_features)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=self.state_shape,
            dtype=np.float32
        )

        # three actions possible - choose each of the three cards
        self.action_space = gym.spaces.Discrete(3)

    def reset(self) -> np.array:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset the state
        super().reset()

        # empty bookkeeping structures
        self.results = []
        self.choices = ([], [])
        self.draft_ordering = list(range(3))

        # reset all agents' internal state
        for agent in self.battle_agents:
            agent.reset()
            agent.seed(self._seed)

        return self._encode_state()

    def step(self, action: Union[int, Action]) -> (np.array, int, bool, dict):
        """Makes an action in the game."""
        # if the draft is finished, there should be no more actions
        if self._draft_is_finished:
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

        # find appropriate value for the provided card index
        if action.origin in (0, 1, 2):
            chosen_index = self.draft_ordering[action.origin]
        else:
            chosen_index = 0

        # find chosen card and keep track of it
        chosen_card = state.current_player.hand[chosen_index]
        self.choices[state.current_player.id].append(chosen_card)

        # execute the action
        state.act(action)

        # init return info
        reward = 0
        done = False
        info = {'phase': state.phase,
                'turn': state.turn,
                'winner': []}

        # if draft is now ended, evaluation should be done
        if self._draft_is_finished:
            # faster evaluation method for when only one battle is required
            # todo: check if this optimization is still necessary
            if self.evaluation_battles == 1:
                winner = self.do_match(state)

                self.results = [1 if winner == PlayerOrder.FIRST else -1]
                info['winner'] = [winner]
            else:
                # for each evaluation battle required, copy the current
                # start-of-battle state and do battle
                for i in range(self.evaluation_battles):
                    state_copy = self.state.clone()

                    winner = self.do_match(state_copy)

                    self.results.append(1 if winner == PlayerOrder.FIRST else -1)
                    info['winner'].append(winner)

            reward = np.mean(self.results)
            done = True

            del info['turn']

        return self._encode_state(), reward, done, info

    def do_match(self, state):
        # reset the agents
        for agent in self.battle_agents:
            agent.reset()

        # while the game doesn't end, get agents acting alternatively
        while state.winner is None:
            agent = self.battle_agents[state.current_player.id]

            action = agent.act(state)

            state.act(action)

        return state.winner

    def _render_text_ended(self):
        if len(self.results) == 1:
            super()._render_text_ended()
        else:
            wins_by_p0 = int(((np.mean(self.results) + 1) / 2) * 100)

            print(f'P0: {wins_by_p0}%; P1: {100 - wins_by_p0}%')

    def _encode_state_draft(self):
        encoded_state = np.full(self.state_shape, 0, dtype=np.float32)

        chosen_cards = self.choices[self.state.current_player.id]

        if not self._draft_is_finished:
            card_choices = self.state.current_player.hand[0:3]

            self.draft_ordering = list(range(3))

            if self.sort_cards:
                sorted_cards = sorted(self.draft_ordering,
                                      key=lambda p: card_choices[p].id)

                self.draft_ordering = list(sorted_cards)

            for i in range(len(card_choices)):
                index = self.draft_ordering[i]

                encoded_state[-(3 - i)] = self.encode_card(card_choices[index])

        if self.use_draft_history:
            if self.sort_cards:
                chosen_cards = sorted(chosen_cards, key=lambda c: c.id)

            for j, card in enumerate(chosen_cards):
                encoded_state[j] = self.encode_card(card)

        return encoded_state

    def _encode_state_battle(self):
        pass


class LOCMDraftSingleEnv(LOCMDraftEnv):
    def __init__(self, draft_agent=RandomDraftAgent(),
                 play_first=True, **kwargs):
        # init the env
        super().__init__(**kwargs)

        # also init the draft agent and the new parameter
        self.draft_agent = draft_agent
        self.play_first = play_first

    def reset(self) -> np.array:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset what is needed
        encoded_state = super().reset()

        # also reset the draft agent
        self.draft_agent.reset()

        return encoded_state

    def step(self, action: Union[int, Action]) -> (np.array, int, bool, dict):
        """Makes an action in the game."""
        # act according to first and second players
        if self.play_first:
            super().step(action)
            result = super().step(self.draft_agent.act(self.state))
        else:
            super().step(self.draft_agent.act(self.state))
            result = super().step(action)

        return result


class LOCMDraftSelfPlayEnv(LOCMDraftEnv):
    def __init__(self, play_first=True, **kwargs):
        # init the env
        super().__init__(**kwargs)

        # also init the new parameters
        self.play_first = play_first
        self.model = None

    def set_model(self, model_builder, env_builder):
        self.model = model_builder(env_builder(self))

    def update_parameters(self, parameters):
        """Update the current parameters in the model with new ones."""
        self.model.load_parameters(parameters, exact_match=True)

    def step(self, action: Union[int, Action]) -> (np.array, int, bool, dict):
        """Makes an action in the game."""
        obs = self._encode_state()

        # act according to first and second players
        if self.play_first:
            super().step(action)
            result = super().step(self.model.predict(obs)[0])
        else:
            super().step(self.model.predict(obs)[0])
            result = super().step(action)

        return result
