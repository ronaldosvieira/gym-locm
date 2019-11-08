from abc import ABC, abstractmethod

import gym
from prettytable import PrettyTable

from gym_locm.engine import Creature, GreenItem, RedItem, BlueItem, State, Phase


class LOCMEnv(gym.Env, ABC):
    card_types = {Creature: 0, GreenItem: 1, RedItem: 2, BlueItem: 3}

    def __init__(self, seed=None):
        self.state = State(seed=seed)

    def seed(self, seed=None):
        """Sets a seed for random choices in the game."""
        self.state.seed(seed)

    def render(self, mode: str = 'text'):
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

    def _render_text_draft(self):
        print(f'######## TURN {self.state.turn}: '
              f'PLAYER {self.state.current_player.id} ########')
        print()

        table = PrettyTable(['Index', 'Name', 'Cost', 'Description'])

        for i, card in enumerate(self.state.current_player.hand):
            table.add_row([i, card.name, card.cost, card.text])

        print(table)

    def _render_text_ended(self):
        print(f'*         *    .            *     .   *      .   *\n'
              f'    .             *   .    * .         .\n'
              f'*        *    .    PLAYER {self.state.winner} WON!       *.   . *\n'
              f'*     .   *         *         .       *.      *   .\n'  
              f'.              *      .     * .         .')

    def _render_text_battle(self):
        pass  # todo: implement

    def _render_native(self):
        return str(self.state)

    @staticmethod
    def encode_card(card):
        """ Encodes a card object into a numerical array. """
        card_type = [1.0 if isinstance(card, card_type) else 0.0
                     for card_type in LOCMEnv.card_types]
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
        """ Encodes a state object into a numerical matrix. """
        if self.state.phase == Phase.DRAFT:
            return self._encode_state_draft()
        elif self.state.phase == Phase.BATTLE:
            return self._encode_state_battle()

    @abstractmethod
    def _encode_state_draft(self):
        """ Encodes a state object in the draft phase. """
        pass

    @abstractmethod
    def _encode_state_battle(self):
        """ Encodes a state object in the battle phase. """
        pass

    @property
    def _draft_is_finished(self):
        return self.state.phase > Phase.DRAFT

    @property
    def _battle_is_finished(self):
        return self.state.phase > Phase.BATTLE
