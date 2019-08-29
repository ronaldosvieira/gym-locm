import gym
from prettytable import PrettyTable

from gym_locm.agents import *
from gym_locm.engine import *


class LoCMDraftEnv(gym.Env):
    metadata = {'render.modes': ['text']}
    card_types = {Creature: 0, GreenItem: 1, RedItem: 2, BlueItem: 3}

    def __init__(self,
                 battle_agents=(RandomBattleAgent(), RandomBattleAgent()),
                 use_draft_history=True,
                 cards_in_deck=30,
                 evaluation_battles=1):
        self.state = Game(cards_in_deck)

        self.battle_agents = battle_agents

        self.results = []
        self.evaluation_battles = evaluation_battles

        self.choices = ([], [])
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

        self.action_space = gym.spaces.Discrete(3)

        self.reset()

    def reset(self):
        self.state = State()
        self.results = []

        return self._encode_state()

    def step(self, action):
        if self._draft_is_finished:
            raise GameIsEndedError()

        if not isinstance(action, Action):
            action = Action(ActionType.PICK, action)

        state = self.state

        chosen_index = action.origin if action.origin is not None else 0
        chosen_card = state.current_player.hand[chosen_index]
        self.choices[state.current_player.id].append(chosen_card)

        state.act(action)

        reward = 0
        done = False
        info = {'phase': state.phase,
                'turn': state.turn,
                'winner': []}

        if self._draft_is_finished:
            if self.evaluation_battles == 1:
                while state.winner is None:
                    agent = self.battle_agents[state.current_player.id]

                    action = agent.act(state)

                    state.act(action)

                if state.winner == PlayerOrder.FIRST:
                    self.results.append(1)
                elif state.winner == PlayerOrder.SECOND:
                    self.results.append(-1)

                info['winner'].append(state.winner)
            else:
                for i in range(self.evaluation_battles):
                    state_copy = copy.deepcopy(self.state)

                    while state_copy.winner is None:
                        agent = self.battle_agents[state_copy.current_player.id]

                        action = agent.act(state_copy)

                        state_copy.act(action)

                    if state_copy.winner == PlayerOrder.FIRST:
                        self.results.append(1)
                    elif state_copy.winner == PlayerOrder.SECOND:
                        self.results.append(-1)

                    info['winner'].append(state_copy.winner)

            reward = np.mean(self.results)
            done = True

            del info['turn']

        return self._encode_state(), reward, done, info

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
        pass  # TODO implement

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

    def render(self, mode='text'):
        if mode == 'text':
            if self.state.phase == Phase.DRAFT:
                self._render_text_draft()
            elif self.state.phase == Phase.BATTLE:
                self._render_text_battle()
            elif self.state.phase == Phase.ENDED:
                self._render_text_ended()

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

            for i, card in enumerate(card_choices):
                encoded_state[-(3 - i)] = self._encode_card(card)

        if self.use_draft_history:
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
                 cards_in_deck=30,
                 evaluation_battles=1,
                 play_first=True):
        super().__init__(battle_agents, use_draft_history,
                         cards_in_deck, evaluation_battles)

        self.draft_agent = draft_agent
        self.play_first = play_first

    def step(self, action):
        if self.play_first:
            super().step(action)
            result = super().step(self.draft_agent.act(self.state))
        else:
            super().step(self.draft_agent.act(self.state))
            result = super().step(action)

        return result
