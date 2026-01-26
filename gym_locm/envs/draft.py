from typing import Union

import gymnasium as gym

from gym_locm.agents import *
from gym_locm.engine import Action
from gym_locm.envs.base_env import LOCMEnv
from gym_locm.envs.rewards import *
from gym_locm.exceptions import *


class LOCMDraftEnv(LOCMEnv):
    metadata = {"render.modes": ["text", "native"]}

    def __init__(
        self,
        battle_agents=(RandomBattleAgent(), RandomBattleAgent()),
        use_draft_history=False,
        use_mana_curve=False,
        sort_cards=False,
        evaluation_battles=1,
        seed=None,
        items=True,
        k=3,
        n=30,
        reward_functions=("win-loss",),
        reward_weights=(1.0,),
        render_mode=None,
    ):
        super().__init__(
            seed=seed,
            version="1.2",
            items=items,
            k=k,
            n=n,
            reward_functions=reward_functions,
            reward_weights=reward_weights,
            render_mode=render_mode,
        )

        # init bookkeeping structures
        self.results = []
        self.choices = ([], [])
        self.draft_ordering = list(range(3))
        self.rewards = [0.0]

        self.battle_agents = battle_agents

        for battle_agent in self.battle_agents:
            battle_agent.seed(seed)

        self.evaluation_battles = evaluation_battles
        self.sort_cards = sort_cards
        self.use_draft_history = use_draft_history
        self.use_mana_curve = use_mana_curve

        self.cards_in_state = self.n + self.k if use_draft_history else self.k
        self.card_features = 16

        self.state_shape = self.cards_in_state * self.card_features

        if self.use_mana_curve:
            self.state_shape += 13

        self.state_shape = (self.state_shape,)

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self.state_shape, dtype=np.float32
        )

        # three actions possible - choose each of the three cards
        self.action_space = gym.spaces.Discrete(3)

        self.reward_range = (-1, 1)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.array, dict]:
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

        self.rewards.append(0.0)

        return self.encode_state(), {}

    def step(
        self, action: Union[int, Action]
    ) -> tuple[np.array, int, bool, bool, dict]:
        """Makes an action in the game."""
        # if the draft is finished, there should be no more actions
        if self._draft_is_finished:
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
        current_player_id = state.current_player.id

        self.last_player_rewards[state.current_player.id] = [
            weight * function.calculate(state, for_player=current_player_id)
            for function, weight in zip(self.reward_functions, self.reward_weights)
        ]

        # find appropriate value for the provided card index
        if 0 <= action.origin < self.k:
            chosen_index = self.draft_ordering[action.origin]
        else:
            chosen_index = 0

        # find chosen card and keep track of it
        chosen_card = state.current_player.hand[chosen_index]
        self.choices[state.current_player.id].append(chosen_card)

        # execute the action
        state.act(action)

        reward_before = self.last_player_rewards[state.current_player.id]
        reward_after = [
            weight * function.calculate(state, for_player=current_player_id)
            for function, weight in zip(self.reward_functions, self.reward_weights)
        ]

        # init return info
        terminated = False
        info = {"phase": state.phase, "turn": state.turn, "winner": []}

        # if draft is now ended, evaluation should be done
        if self._draft_is_finished:
            # faster evaluation method for when only one battle is required
            # todo: check if this optimization is still necessary
            if self.evaluation_battles == 1:
                winner = self.do_match(state)

                self.results = [1 if winner == PlayerOrder.FIRST else -1]
                info["winner"] = [winner]
            else:
                # for each evaluation battle required, copy the current
                # start-of-battle state and do battle
                for i in range(self.evaluation_battles):
                    state_copy = self.state.clone()

                    winner = self.do_match(state_copy)

                    self.results.append(1 if winner == PlayerOrder.FIRST else -1)
                    info["winner"].append(winner)

            try:
                win_loss_reward_index = list(map(type, self.reward_functions)).index(
                    WinLossRewardFunction
                )
                reward_after[win_loss_reward_index] = np.mean(self.results)
            except ValueError:
                pass

            terminated = True

        if reward_before is None:
            raw_rewards = (0.0,) * len(self.reward_functions)
        else:
            raw_rewards = tuple(
                [after - before for before, after in zip(reward_before, reward_after)]
            )

        info["raw_rewards"] = raw_rewards
        reward = sum(raw_rewards)

        self.rewards[-1] += reward

        return self.encode_state(), reward, terminated, False, info

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

            print(f"P0: {wins_by_p0}%; P1: {100 - wins_by_p0}%")

    def _encode_state_deck_building(self):
        encoded_state = np.full(self.state_shape, 0, dtype=np.float32)

        chosen_cards = self.choices[self.state.current_player.id]

        if not self._draft_is_finished:
            card_choices = self.state.current_player.hand[0 : self.k]

            self.draft_ordering = list(range(self.k))

            if self.sort_cards:
                sorted_cards = sorted(
                    self.draft_ordering, key=lambda p: card_choices[p].id
                )

                self.draft_ordering = list(sorted_cards)

            for i in range(len(card_choices)):
                index = self.draft_ordering[i]
                lo = -(self.k - i) * self.card_features
                hi = lo + self.card_features
                hi = hi if hi < 0 else None

                encoded_state[lo:hi] = self.encode_card(
                    card_choices[index], version="1.2"
                )

        if self.use_draft_history:
            if self.sort_cards:
                chosen_cards = sorted(chosen_cards, key=lambda c: c.id)

            for j, card in enumerate(chosen_cards):
                lo = -(self.n + self.k - j) * self.card_features
                hi = lo + self.card_features
                hi = hi if hi < 0 else None

                encoded_state[lo:hi] = self.encode_card(card, version="1.2")

        if self.use_mana_curve:
            for chosen_card in chosen_cards:
                encoded_state[chosen_card.cost] += 1

        return encoded_state

    def _encode_state_battle(self):
        pass

    def get_episode_rewards(self):
        return self.rewards


class LOCMDraftSingleEnv(LOCMDraftEnv):
    def __init__(self, draft_agent=RandomDraftAgent(), play_first=True, **kwargs):
        # init the env
        super().__init__(**kwargs)

        # also init the draft agent and the new parameter
        self.draft_agent = draft_agent
        self.play_first = play_first

        self.rewards_single_player = []

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.array, dict]:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset what is needed
        encoded_state = super().reset()

        # also reset the draft agent
        self.draft_agent.reset()

        self.rewards_single_player.append(0.0)

        return encoded_state, {}

    def step(
        self, action: Union[int, Action]
    ) -> tuple[np.array, int, bool, bool, dict]:
        """Makes an action in the game."""
        # act according to first and second players
        if self.play_first:
            super().step(action)
            state, reward, terminated, truncated, info = super().step(
                self.draft_agent.act(self.state)
            )
        else:
            super().step(self.draft_agent.act(self.state))
            state, reward, terminated, truncated, info = super().step(action)
            reward = -reward

        try:
            self.rewards_single_player[-1] += reward
        except IndexError:
            self.rewards_single_player = [reward]

        return state, reward, terminated, truncated, info

    def get_episode_rewards(self):
        return self.rewards_single_player


class LOCMDraftSelfPlayEnv(LOCMDraftEnv):
    def __init__(self, play_first, adversary_policy=None, **kwargs):
        # init the env
        super().__init__(**kwargs)

        # also init the new parameters
        self.play_first = play_first
        self.adversary_policy = adversary_policy

        self.rewards_single_player = []

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.array, dict]:
        encoded_state = super().reset()

        self.rewards_single_player.append(0.0)

        return encoded_state, {}

    def step(
        self, action: Union[int, Action]
    ) -> tuple[np.array, int, bool, bool, dict]:
        """Makes an action in the game."""
        obs = self.encode_state()

        # act according to first and second players
        if self.play_first:
            super().step(action)
            state, reward, terminated, truncated, info = super().step(
                self.adversary_policy(obs)
            )
        else:
            super().step(self.adversary_policy(obs))
            state, reward, terminated, truncated, info = super().step(action)
            reward = -reward

        try:
            self.rewards_single_player[-1] += reward
        except IndexError:
            self.rewards_single_player = [reward]

        return state, reward, terminated, truncated, info

    def get_episode_rewards(self):
        return self.rewards_single_player
