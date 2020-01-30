from abc import ABC, abstractmethod
from operator import attrgetter
from sty import fg

import gym
from prettytable import PrettyTable

from gym_locm.engine import Creature, GreenItem, RedItem, BlueItem, State, Phase, ActionType, Action, Lane
from gym_locm.exceptions import MalformedActionError


class LOCMEnv(gym.Env, ABC):
    card_types = {Creature: 0, GreenItem: 1, RedItem: 2, BlueItem: 3}

    def __init__(self, seed=None):
        self._seed = seed

        self.state = State(seed=seed)

    def seed(self, seed=None):
        """Sets a seed for random choices in the game."""
        self.state.seed(seed)

    def reset(self):
        """
        Resets the environment.
        The game is put into its initial state
        """
        if self._seed is None:
            # recover random state from current state obj
            random_state = self.state.np_random

            # start a brand new game
            self.state = State()

            # apply random state
            self.state.np_random = random_state
        else:
            # start a brand new game with next seed
            self._seed += 1

            self.state = State(seed=self._seed)

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
        # if ascii mode, print appropriate representation
        if mode == 'ascii':
            if self.state.phase == Phase.DRAFT:
                self._render_ascii_draft()
            elif self.state.phase == Phase.BATTLE:
                pass  # todo: implement
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
        player = self.state.current_player
        opponent = self.state.opposing_player

        print(f'######## TURN {self.state.turn}: '
              f'PLAYER {player.id} ########')
        print()
        print("Stats:")
        print(f"{player.health} HP, {player.mana}/{player.base_mana} MP")
        print(f"Next rune: {player.next_rune}, "
              f"next draw: {1 + player.bonus_draw}")
        print()

        print("Hand:")

        table = PrettyTable(['Id', 'Name', 'Cost', 'Description'])

        for i, card in enumerate(sorted(player.hand, key=attrgetter('cost'))):
            table.add_row([card.instance_id, card.name, card.cost, card.text])

        print(table)
        print()
        print("Board:")

        table = PrettyTable(['Id', 'Name', 'Lane',
                             'Stats', 'Can attack?'])

        for lane, cards in zip(['Left', 'Right'], player.lanes):
            for card in cards:
                card_text = f"{card.attack}/{card.defense} "
                card_text += f"{''.join(card.keywords)}"

                table.add_row([card.instance_id, card.name, lane, card_text,
                               'Yes' if card.able_to_attack() else 'No'])

        print(table)
        print()
        print("Opponent's stats:")
        print(f"{opponent.health} HP, {opponent.mana}/{opponent.base_mana} MP")
        print(f"Next rune: {opponent.next_rune}, "
              f"next draw: {1 + opponent.bonus_draw}")
        print(f"Cards in hand: {len(opponent.hand)}")
        print()

        last_actions = []

        for action in reversed(opponent.actions[:-1]):
            if action.type == ActionType.PASS:
                break

            last_actions.append(action)

        print("Last actions:")

        if last_actions:
            for a in reversed(last_actions):
                target_id = -1 if a.target is None else a.target

                print(f"{a.resolved_origin.id} {a.type.name} "
                      f"{a.origin} {target_id}")
        else:
            print("(none)")

        print()

        print("Opponent's board:")

        table = PrettyTable(['Id', 'Name', 'Lane', 'Stats'])

        for lane, cards in zip(['Left', 'Right'], opponent.lanes):
            for card in cards:
                card_text = f"{card.attack}/{card.defense} "
                card_text += f"{''.join(card.keywords)}"

                table.add_row([card.instance_id, card.name, lane, card_text])

        print(table)

    def _render_ascii_draft(self):
        card_template = [
            '+---------+',
            '|         |',
            '|         |',
            '|         |',
            '+---{}---+',
            '| {} {} |',
            '| {} |',
            '| {} {} {} |',
            '+---------+'
        ]

        hand = self.state.current_player.hand

        cards_ascii = []

        for i, card in enumerate(hand):
            card_ascii = list(card_template)

            cost = f"{{{card.cost}}}" if card.cost < 10 else card.cost
            attack = format(str(card.attack), '<3')
            defense = format(str(card.defense), '>3')
            keywords = "".join(a if card.has_ability(a) else " " for a in 'BCDXGLW')
            player_hp, enemy_hp, card_draw = "  ", "  ", " "

            if card.player_hp > 0:
                player_hp = f"+{card.player_hp}"
            elif card.player_hp < 0:
                player_hp = card.player_hp

            if card.enemy_hp > 0:
                enemy_hp = f"+{card.enemy_hp}"
            elif card.enemy_hp < 0:
                enemy_hp = card.enemy_hp

            if card.card_draw > 0:
                card_draw = str(card.card_draw)

            colors = {Creature: fg.li_yellow, GreenItem: fg.li_green,
                      RedItem: fg.li_red, BlueItem: fg.li_blue}
            color = colors[type(card)]

            name = format(card.name[:27], '<27')

            card_ascii[1] = '|' + name[:9] + '|'
            card_ascii[2] = '|' + name[9:18] + '|'
            card_ascii[3] = '|' + name[18:27] + '|'
            card_ascii[4] = card_ascii[4].format(cost)
            card_ascii[5] = card_ascii[5].format(attack, defense)
            card_ascii[6] = card_ascii[6].format(keywords)
            card_ascii[7] = card_ascii[7].format(player_hp, enemy_hp, card_draw)

            card_ascii = list(map(lambda l: color + l + fg.rs, card_ascii))

            cards_ascii.append(card_ascii)

        for line in zip(*cards_ascii):
            print(" ".join(line))

        print("  card 0  ", "  card 1  ", "  card 2  ")

    def _render_native(self):
        return str(self.state)

    def decode_action(self, action_number):
        """
        Decodes an action number from either phases into the
        corresponding action object, if possible. Raises
        MalformedActionError otherwise.
        """
        try:
            if self.state.phase == Phase.DRAFT:
                return self.decode_draft_action(action_number)
            elif self.state.phase == Phase.BATTLE:
                return self.decode_battle_action(action_number)
        finally:
            return None

    def decode_draft_action(self, action_number):
        """
        Decodes an action number (0-2) from draft phase into the
        corresponding action object, if possible. Raises
        MalformedActionError otherwise.
        """

        if action_number not in (0, 1, 2):
            raise MalformedActionError("Invalid action number")

        return Action(ActionType.PICK, action_number)

    def decode_battle_action(self, action_number):
        """
        Decodes an action number (0-162) from battle phase into
        the corresponding action object, if possible. Raises
        MalformedActionError otherwise.
        """
        player = self.state.current_player
        opponent = self.state.opposing_player

        try:
            if action_number == 0:
                return Action(ActionType.PASS)
            elif 1 <= action_number <= 16:
                action_number -= 1

                origin = int(action_number / 2)
                target = Lane(action_number % 2)

                origin = player.hand[origin].instance_id

                return Action(ActionType.SUMMON, origin, target)
            elif 17 <= action_number <= 120:
                action_number -= 17

                origin = int(action_number / 13)
                target = action_number % 13

                origin = player.hand[origin].instance_id

                if target == 0:
                    target = None
                else:
                    target -= 1

                    side = [player, opponent][int(target / 6)]
                    lane = int((target % 6) / 3)
                    index = target % 3

                    target = side.lanes[lane][index].instance_id

                return Action(ActionType.USE, origin, target)
            elif 121 <= action_number <= 162:
                action_number -= 121

                origin = int(action_number / 7)
                target = action_number % 7

                origin = player.lanes[int(origin / 3)][origin % 3].instance_id

                if target == 0:
                    target = None
                else:
                    target -= 1

                    target = opponent.lanes[int(target / 3)][target % 3].instance_id

                return Action(ActionType.ATTACK, origin, target)
            else:
                raise MalformedActionError("Invalid action number")
        except IndexError:
            raise MalformedActionError("Invalid action number")

    @staticmethod
    def encode_card(card):
        """Encodes a card object into a numerical array."""
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

    @staticmethod
    def encode_card_on_board(card):
        """Encodes a card object into a numerical array."""
        attack = card.attack / 12
        defense = card.defense / 12
        keywords = list(map(int, map(card.keywords.__contains__, 'BCDGLW')))

        return [attack, defense] + keywords

    @staticmethod
    def encode_players(current, opposing):
        return current.health, current.mana, current.next_rune, \
               1 + current.bonus_draw, opposing.health, \
               opposing.base_mana + opposing.bonus_mana, \
               opposing.next_rune, 1 + opposing.bonus_draw

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
