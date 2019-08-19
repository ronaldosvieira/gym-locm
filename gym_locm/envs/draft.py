import gym
from prettytable import PrettyTable

from gym_locm.agents import *
from gym_locm.engine import *


class LoCMDraftEnv(gym.Env):
    metadata = {'render.modes': ['text']}
    card_types = {Creature: 0, GreenItem: 1, RedItem: 2, BlueItem: 3}

    def __init__(self,
                 battle_agent=RandomBattleAgent(),
                 use_draft_history=True,
                 cards_in_deck=30):
        self.state = None
        self.turn = 1
        self.battle_agent = battle_agent

        self.cards_in_deck = cards_in_deck

        self.game = Game()

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
        self.turn = 1

        self.state = self.game.reset()

        return self._encode_state()

    def step(self, action):
        if not isinstance(action, Action):
            action = Action(ActionType.PICK, action)

        new_state, done, info = self.game.step(action)

        reward = 0

        if info['phase'] == Phase.DRAFT:
            self.turn = info['turn']

            self.state = new_state
        elif info['phase'] == Phase.BATTLE:
            current_player = self.state.current_player

            chosen_index = action.origin if action.origin is not None else 0
            chosen_card = current_player.hand[chosen_index]
            current_player.deck.append(chosen_card)

            for player in self.state.players:
                player.hand = []

            while not done:
                action = self.battle_agent.act(new_state)

                new_state, done, info = self.game.step(action)

        if info['phase'] >= Phase.BATTLE:
            if info['winner'] == PlayerOrder.FIRST:
                reward = 1
            elif info['winner'] == PlayerOrder.SECOND:
                reward = -1

            self.state.current_phase = Phase.ENDED

        return self._encode_state(), reward, done, info

    def _render_text_draft(self):
        playing_first = len(self.state.current_player.deck) == \
                 len(self.state.opposing_player.deck)
        print(f'######## TURN {self.turn} ########')
        print()
        print(f"Choosing for player {0 if playing_first else 1}")

        table = PrettyTable(['Index', 'Name', 'Cost', 'Description'])

        for i, card in enumerate(self.state.current_player.hand):
            table.add_row([i, card.name, card.cost, card.text])

        print(table)

    def _render_text_battle(self):
        pass  # TODO implement

    def _render_text_ended(self):
        winner = 1 if self.state.players[0].health <= 0 else 0

        print(f'*         *    .            *     .   *      .   *\n'
              f'    .             *   .    * .         .\n'
              f'*        *    .    PLAYER {winner} WON!       *.   . *\n'
              f'*     .   *         *         .       *.      *   .\n'  
              f'.              *      .     * .         .')

    def render(self, mode='text'):
        if mode == 'text':
            if self.state.current_phase == Phase.DRAFT:
                self._render_text_draft()
            elif self.state.current_phase == Phase.BATTLE:
                self._render_text_battle()
            elif self.state.current_phase == Phase.ENDED:
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

        card_choices = self.state.current_player.hand[0:3]
        chosen_cards = self.state.current_player.deck

        for i, card in enumerate(card_choices):
            encoded_state[-(3 - i)] = self._encode_card(card)

        if self.use_draft_history:
            for j, card in enumerate(chosen_cards):
                encoded_state[j] = self._encode_card(card)

        return encoded_state


class LoCMDraftSingleEnv(LoCMDraftEnv):
    def __init__(self, battle_agent=RandomBattleAgent(),
                 draft_agent=RandomDraftAgent(),
                 use_draft_history=True,
                 cards_in_deck=30,
                 play_first=True):
        super().__init__(battle_agent, use_draft_history, cards_in_deck)

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
