from typing import Union

import gym

from gym_locm.agents import *
from gym_locm.engine import *
from gym_locm.envs.base_env import LOCMEnv


class LOCMConstructedEnv(LOCMEnv):
    metadata = {'render.modes': ['text', 'native']}

    def __init__(self,
                 battle_agents=(RandomBattleAgent(), RandomBattleAgent()),
                 evaluation_battles=1,
                 seed=None, items=True, n=30):
        super().__init__(seed=seed, items=items, n=n, mode='constructed')

        # init bookkeeping structures
        self.results = []

        self.battle_agents = battle_agents

        for battle_agent in self.battle_agents:
            battle_agent.seed(seed)

        self.evaluation_battles = evaluation_battles

        self.cards_in_state = 60
        self.card_features = 16
        self.state_shape = self.cards_in_state * self.card_features

        self.state_shape = self.state_shape,

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=self.state_shape,
            dtype=np.float32
        )

        # three actions possible - choose each of the three cards
        self.action_space = gym.spaces.Discrete(60)

        self.reward_range = (-1, 1)

    def reset(self) -> np.array:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset the state
        super().reset()

        # empty bookkeeping structures
        self.results = []

        # reset all agents' internal state
        for agent in self.battle_agents:
            agent.reset()
            agent.seed(self._seed)

        return self.encode_state()

    def step(self, action: Union[int, Action]) -> (np.array, int, bool, dict):
        """Makes an action in the game."""
        # if the draft is finished, there should be no more actions
        if self._constructed_is_finished:
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
        if 0 <= action.origin < 60:
            chosen_index = action.origin
        else:
            raise MalformedActionError("Invalid action: card to be picked "
                                       "should be between 0 and 59")

        # execute the action
        state.act(action)

        # init return info
        reward = 0
        done = False
        info = {'phase': state.phase,
                'turn': state.turn,
                'winner': []}

        # if constructed is now ended, evaluation should be done
        if self._constructed_is_finished:
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

        return self.encode_state(), reward, done, info

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

    def _encode_state_constructed(self):
        encoded_state = np.full(self.state_shape, 0, dtype=np.float32)

        if not self._constructed_is_finished:
            card_choices = self.state._constructed_cards

            for i in range(len(card_choices)):
                lo = -(60 - i) * self.card_features
                hi = lo + self.card_features
                hi = hi if hi < 0 else None

                encoded_state[lo:hi] = self.encode_card(card_choices[i])

        return encoded_state

    def _encode_state_draft(self):
        pass

    def _encode_state_battle(self):
        pass


class LOCMConstructedSingleEnv(LOCMConstructedEnv):
    def __init__(self, constructed_agent=RandomDraftAgent(),
                 play_first=True, **kwargs):
        # init the env
        super().__init__(**kwargs)

        # also init the constructed agent and the new parameter
        self.constructed_agent = constructed_agent
        self.play_first = play_first

    def reset(self) -> np.array:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset what is needed
        encoded_state = super().reset()

        # also reset the draft agent
        self.constructed_agent.reset()

        return encoded_state

    def step(self, action: Union[int, Action]) -> (np.array, int, bool, dict):
        """Makes an action in the game."""
        # act according to first and second players
        if self.play_first:
            super().step(action)
            state, reward, done, info = super().step(self.constructed_agent.act(self.state))
        else:
            super().step(self.constructed_agent.act(self.state))
            state, reward, done, info = super().step(action)
            reward = -reward

        return state, reward, done, info


class LOCMConstructedSelfPlayEnv(LOCMConstructedEnv):
    def __init__(self, play_first, adversary_policy=None, **kwargs):
        # init the env
        super().__init__(**kwargs)

        # also init the new parameters
        self.play_first = play_first
        self.adversary_policy = adversary_policy

    def step(self, action: Union[int, Action]) -> (np.array, int, bool, dict):
        """Makes an action in the game."""
        obs = self.encode_state()

        # act according to first and second players
        if self.play_first:
            super().step(action)
            state, reward, done, info = super().step(self.adversary_policy(obs))
        else:
            super().step(self.adversary_policy(obs))
            state, reward, done, info = super().step(action)
            reward = -reward

        return state, reward, done, info

