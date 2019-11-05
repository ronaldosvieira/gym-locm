from typing import Union

import gym
from prettytable import PrettyTable

from gym_locm.agents import *
from gym_locm.engine import *


class LoCMDraftEnv(gym.Env):
    metadata = {'render.modes': ['text', 'native']}
    card_types = {Creature: 0, GreenItem: 1, RedItem: 2, BlueItem: 3}

    def __init__(self,
                 battle_agents=(RandomBattleAgent(), RandomBattleAgent()),
                 use_draft_history=True,
                 sort_cards=True,
                 evaluation_battles=1,
                 seed=None):
        # init bookkeeping structures
        self.results = []
        self.choices = ([], [])
        self.draft_ordering = list(range(3))

        self.battle_agents = battle_agents
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

        # init game
        self.state = State(seed=seed)

    def seed(self, seed=None):
        """Sets a seed for random choices in the game."""
        self.state.seed(seed)

    def reset(self) -> np.array:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # recover random state from current state obj
        random_state = self.state.np_random

        # start a brand new game
        self.state = State()

        # apply random state
        self.state.np_random = random_state

        # empty bookkeeping structures
        self.results = []
        self.choices = ([], [])
        self.draft_ordering = list(range(3))

        # reset all agents' internal state
        for agent in self.battle_agents:
            agent.reset()

        return self._encode_state()

    def step(self, action: Union[int, Action]) -> (np.array, int, bool, dict):
        """Makes an action in the game."""
        # if the draft is finished, there should be no more actions
        if self._draft_is_finished:
            raise GameIsEndedError()

        # if it's an integer, wrap it in an action object
        if not isinstance(action, Action):
            action = Action(ActionType.PICK, action)

        # less property accesses
        state = self.state

        # find appropriate value for the provided card index
        if action.origin is not None:
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

    def _render_text_draft(self):
        playing_first = len(self.state.current_player.deck) == \
                 len(self.state.opposing_player.deck)
        print(f'######## TURN {self.state.turn} ########')
        print()
        print(f"Choosing for player {0 if playing_first else 1}")

        table = PrettyTable(['Index', 'Name', 'Cost', 'Description'])

        for i, card in enumerate(self.state.current_player.hand):
            table.add_row([i, card.name, card.cost, card.text])

        print(table)

    def _render_text_battle(self):
        pass  # todo: implement

    def _render_text_ended(self):
        if len(self.results) == 1:
            print(f'*         *    .            *     .   *      .   *\n'
                  f'    .             *   .    * .         .\n'
                  f'*        *    .    PLAYER {self.state.winner} WON!       *.   . *\n'
                  f'*     .   *         *         .       *.      *   .\n'  
                  f'.              *      .     * .         .')
        else:
            wins_by_p0 = int(((np.mean(self.results) + 1) / 2) * 100)

            print(f'P0: {wins_by_p0}%; P1: {100 - wins_by_p0}%')

    def _render_native(self):
        return str(self.state)

    def render(self, mode: str = 'text') -> Union[None, str]:
        """Builds a representation of the current state."""
        # if text mode, print appropriate representation
        if mode == 'text':
            if self.state.phase == Phase.DRAFT:
                self._render_text_draft()
            elif self.state.phase == Phase.BATTLE:
                self._render_text_battle()
            elif self.state.phase == Phase.ENDED:
                self._render_text_ended()
        # if native mode, build and return input string
        elif mode == 'native':
            return self._render_native()

    def _encode_card(self, card):
        card_type = [1.0 if isinstance(card, card_type) else 0.0
                     for card_type in self.card_types]
        cost = card.cost / 12
        attack = card.attack / 12
        defense = max(-12, card.defense) / 12
        keywords = list(map(int, map(card.keywords.__contains__, 'BCDGLW')))
        player_hp = card.player_hp / 12
        enemy_hp = card.enemy_hp / 12
        card_draw = card.card_draw / 2

        return card_type + [cost, attack, defense, player_hp,
                            enemy_hp, card_draw] + keywords

    def _encode_state(self):
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

                encoded_state[-(3 - i)] = self._encode_card(card_choices[i])

        if self.use_draft_history:
            if self.sort_cards:
                chosen_cards = sorted(chosen_cards, key=lambda c: c.id)

            for j, card in enumerate(chosen_cards):
                encoded_state[j] = self._encode_card(card)

        return encoded_state

    @property
    def _draft_is_finished(self):
        return self.state.phase != Phase.DRAFT


class LoCMDraftSingleEnv(LoCMDraftEnv):
    def __init__(self, battle_agents=(RandomBattleAgent(), RandomBattleAgent()),
                 draft_agent=RandomDraftAgent(),
                 use_draft_history=True,
                 sort_cards=True,
                 evaluation_battles=1,
                 seed=None,
                 play_first=True):
        # init the env
        super().__init__(battle_agents, use_draft_history, sort_cards,
                         evaluation_battles, seed)

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


class LoCMDraftSelfPlayEnv(LoCMDraftEnv):
    def __init__(self, model,
                 battle_agents=(RandomBattleAgent(), RandomBattleAgent()),
                 use_draft_history=True,
                 sort_cards=True,
                 evaluation_battles=1,
                 seed=None,
                 play_first=True):
        # init the env
        super().__init__(battle_agents, use_draft_history, sort_cards,
                         evaluation_battles, seed)

        # also init the new parameters
        self.play_first = play_first
        self.model = model

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
