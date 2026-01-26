from abc import ABC, abstractmethod
from operator import attrgetter

import numpy as np
import gymnasium as gym

try:
    from sty import fg
    from prettytable import PrettyTable
except ImportError:
    pass

from gym_locm.engine import (
    Creature,
    GreenItem,
    RedItem,
    BlueItem,
    State,
    Phase,
    ActionType,
    Action,
    Lane,
    Area,
)
from gym_locm.envs.rewards import parse_reward
from gym_locm.exceptions import MalformedActionError


class LOCMEnv(gym.Env, ABC):
    card_types = {Creature: 0, GreenItem: 1, RedItem: 2, BlueItem: 3}

    def __init__(
        self,
        seed=None,
        version="1.5",
        items=True,
        k=None,
        n=30,
        reward_functions=("win-loss",),
        reward_weights=(1.0,),
        render_mode="native",
    ):
        self._seed = seed
        self.render_mode = render_mode
        self.version = version
        self.episodes = 0
        self.items = items

        if version == "1.5":
            k = k if k is not None else 120
        else:
            k = k if k is not None else 3

        self.k, self.n = k, n

        assert len(reward_functions) == len(
            reward_weights
        ), "The length of reward_functions and reward_weights must be the same"

        self.reward_functions = tuple(
            [parse_reward(function_name)() for function_name in reward_functions]
        )
        self.reward_weights = reward_weights

        self.last_player_rewards = [None, None]

        self.reward_range = (-sum(reward_weights), sum(reward_weights))

        self.state = State(
            seed=seed, items=items, version=version, deck_building_kwargs=dict(k=k, n=n)
        )

    def seed(self, seed=None):
        """Sets a seed for random choices in the game."""
        self._seed = seed
        self.state.seed(seed)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.array, dict]:
        """
        Resets the environment.
        The game is put into its initial state
        """
        super().reset(seed=seed)

        if seed:
            self.seed(seed)

        if self._seed is None:
            # recover random state from current state obj
            rng = self.state.rng

            # start a brand new game
            self.state = State(
                items=self.items,
                version=self.version,
                deck_building_kwargs=dict(k=self.k, n=self.n),
            )

            # apply random state
            self.state.rng = rng
        else:
            # start a brand new game with next seed
            self._seed += 1

            self.state = State(
                seed=self._seed,
                items=self.items,
                version=self.version,
                deck_building_kwargs=dict(k=self.k, n=self.n),
            )

        self.episodes += 1
        self.last_player_rewards = [None, None]

        return self.encode_state(), {}

    def render(self, mode: str | None = None):
        """Builds a representation of the current state."""
        if mode is None:
            mode = self.render_mode

        # if text mode, print appropriate representation
        if mode == "text":
            if self.state.phase == Phase.DECK_BUILDING:
                self._render_text_deck_building()
            elif self.state.phase == Phase.BATTLE:
                self._render_text_battle()
            elif self.state.phase == Phase.ENDED:
                self._render_text_ended()
        # if ascii mode, print appropriate representation
        if mode == "ascii":
            if self.state.phase == Phase.DECK_BUILDING and self.version == "1.2":
                self._render_ascii_draft()
            elif self.state.phase == Phase.DECK_BUILDING and self.version == "1.5":
                pass  # todo: implement
            elif self.state.phase == Phase.BATTLE:
                pass  # todo: implement
            elif self.state.phase == Phase.ENDED:
                self._render_text_ended()
        # if native mode, build and return input string
        elif mode == "native":
            return self._render_native()

    def _render_text_deck_building(self):
        print(
            f"######## TURN {self.state.turn}: "
            f"PLAYER {self.state.current_player.id} ########"
        )
        print()

        try:
            table = PrettyTable(["Index", "Name", "Cost", "Description"])
        except NameError:
            raise ImportError(
                "To use the 'text' and 'ascii' rendering modes, please install `gym-locm[rendering]`."
            )

        for i, card in enumerate(self.state.current_player.hand):
            table.add_row([i, card.name, card.cost, card.text])

        print(table)

    def _render_text_ended(self):
        print(
            f"*         *    .            *     .   *      .   *\n"
            f"    .             *   .    * .         .\n"
            f"*        *    .    PLAYER {self.state.winner} WON!       *.   . *\n"
            f"*     .   *         *         .       *.      *   .\n"
            f".              *      .     * .         ."
        )

    def _render_text_battle(self):
        player = self.state.current_player
        opponent = self.state.opposing_player

        print(f"######## TURN {self.state.turn}: " f"PLAYER {player.id} ########")
        print()
        print("Stats:")
        print(f"{player.health} HP, {player.mana}/{player.base_mana} MP")
        print(f"Next rune: {player.next_rune}, " f"next draw: {1 + player.bonus_draw}")
        print()

        print("Hand:")

        if self.version == "1.5":
            table_headers = ["Id", "Name", "Cost", "Area", "Description"]
        else:
            table_headers = ["Id", "Name", "Cost", "Description"]

        try:
            table = PrettyTable(table_headers)
        except NameError:
            raise ImportError(
                "To use the 'text' and 'ascii' rendering modes, please install `gym-locm[rendering]`."
            )

        for i, card in enumerate(sorted(player.hand, key=attrgetter("cost"))):
            if self.version == "1.5":
                table.add_row(
                    [
                        card.instance_id,
                        card.name,
                        card.cost,
                        Area(card.area).name,
                        card.text,
                    ]
                )
            else:
                table.add_row([card.instance_id, card.name, card.cost, card.text])

        print(table)
        print()
        print("Board:")

        table = PrettyTable(["Id", "Name", "Lane", "Stats", "Can attack?"])

        for lane, cards in zip(["Left", "Right"], player.lanes):
            for card in cards:
                card_text = f"{card.attack}/{card.defense} "
                card_text += f"{''.join(card.keywords)}"

                table.add_row(
                    [
                        card.instance_id,
                        card.name,
                        lane,
                        card_text,
                        "Yes" if card.able_to_attack() else "No",
                    ]
                )

        print(table)
        print()
        print("Opponent's stats:")
        print(f"{opponent.health} HP, {opponent.mana}/{opponent.base_mana} MP")
        print(
            f"Next rune: {opponent.next_rune}, " f"next draw: {1 + opponent.bonus_draw}"
        )
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

                print(
                    f"{a.resolved_origin.id} {a.type.name} " f"{a.origin} {target_id}"
                )
        else:
            print("(none)")

        print()

        print("Opponent's board:")

        table = PrettyTable(["Id", "Name", "Lane", "Stats"])

        for lane, cards in zip(["Left", "Right"], opponent.lanes):
            for card in cards:
                card_text = f"{card.attack}/{card.defense} "
                card_text += f"{''.join(card.keywords)}"

                table.add_row([card.instance_id, card.name, lane, card_text])

        print(table)

    def _render_ascii_draft(self):
        card_template = [
            "+---------+",
            "|         |",
            "|         |",
            "|         |",
            "+---{}---+",
            "| {} {} |",
            "| {} |",
            "| {} {} {} |",
            "+---------+",
        ]

        hand = self.state.current_player.hand

        cards_ascii = []

        for i, card in enumerate(hand):
            card_ascii = list(card_template)

            cost = f"{{{card.cost}}}" if card.cost < 10 else card.cost
            attack = format(str(card.attack), "<3")
            defense = format(str(card.defense), ">3")
            keywords = "".join(a if card.has_ability(a) else " " for a in "BCDXGLW")
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

            try:
                colors = {
                    Creature: fg.li_yellow,
                    GreenItem: fg.li_green,
                    RedItem: fg.li_red,
                    BlueItem: fg.li_blue,
                }
            except NameError:
                raise ImportError(
                    "To use the 'text' and 'ascii' rendering modes, please install `gym-locm[rendering]`."
                )

            color = colors[type(card)]

            name = format(card.name[:27], "<27")

            card_ascii[1] = "|" + name[:9] + "|"
            card_ascii[2] = "|" + name[9:18] + "|"
            card_ascii[3] = "|" + name[18:27] + "|"
            card_ascii[4] = card_ascii[4].format(cost)
            card_ascii[5] = card_ascii[5].format(attack, defense)
            card_ascii[6] = card_ascii[6].format(keywords)
            card_ascii[7] = card_ascii[7].format(player_hp, enemy_hp, card_draw)

            card_ascii = list(map(lambda l: color + l + fg.rs, card_ascii))

            cards_ascii.append(card_ascii)

        for line in zip(*cards_ascii):
            print(" ".join(line))

        for i in range(len(hand)):
            print(f"  card {i}  ", end=" ")

    def _render_native(self):
        return str(self.state)

    def decode_action(self, action_number):
        """
        Decodes an action number from either phases into the
        corresponding action object, if possible. Raises
        MalformedActionError otherwise.
        """
        try:
            if self.state.phase == Phase.DECK_BUILDING:
                return self.decode_deck_building_action(action_number)
            elif self.state.phase == Phase.BATTLE:
                return self.decode_battle_action(action_number)
            else:
                return None
        except MalformedActionError:
            return None

    def decode_deck_building_action(self, action_number):
        """
        Decodes an action number (0, k-1) from deck-building phase
        into the corresponding action object, if possible. Raises
        MalformedActionError otherwise.
        """

        if action_number < 0 or action_number >= self.k:
            raise MalformedActionError("Invalid action number")

        return Action(
            ActionType.CHOOSE if self.version == "1.5" else ActionType.PICK,
            action_number,
        )

    def decode_battle_action(self, action_number):
        """
        Decodes an action number (0-144) from battle phase into
        the corresponding action object, if possible. Raises
        MalformedActionError otherwise.
        """
        player = self.state.current_player
        opponent = self.state.opposing_player

        if not self.items and action_number > 16:
            action_number += 104

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
            elif 121 <= action_number <= 144:
                action_number -= 121

                origin = action_number // 4
                target = action_number % 4

                lane = origin // 3

                origin = player.lanes[lane][origin % 3].instance_id

                if target == 0:
                    target = None
                else:
                    target -= 1

                    target = opponent.lanes[lane][target].instance_id

                return Action(ActionType.ATTACK, origin, target)
            else:
                raise MalformedActionError("Invalid action number")
        except IndexError:
            raise MalformedActionError("Invalid action number")

    @staticmethod
    def encode_card(card, version="1.2"):
        """Encodes a card object into a numerical array."""
        card_type = [
            1.0 if isinstance(card, card_type) else 0.0
            for card_type in LOCMEnv.card_types
        ]
        cost = card.cost / 12
        attack = card.attack / 12
        defense = max(-12, card.defense) / 12
        keywords = list(map(int, map(card.keywords.__contains__, "BCDGLW")))
        player_hp = card.player_hp / 12
        enemy_hp = card.enemy_hp / 12
        card_draw = card.card_draw / 2

        encoded_card = (
            card_type
            + [cost, attack, defense, player_hp, enemy_hp, card_draw]
            + keywords
        )

        if version == "1.5":
            encoded_card += [card.area / 2]

        return encoded_card

    @staticmethod
    def encode_friendly_card_on_board(card: Creature):
        """Encodes a card object into a numerical array."""
        attack = card.attack / 12
        defense = card.defense / 12
        can_attack = int(card.can_attack and not card.has_attacked_this_turn)
        keywords = list(map(int, map(card.keywords.__contains__, "BCDGLW")))

        return [attack, defense, can_attack] + keywords

    @staticmethod
    def encode_enemy_card_on_board(card: Creature):
        """Encodes a card object into a numerical array."""
        attack = card.attack / 12
        defense = card.defense / 12
        keywords = list(map(int, map(card.keywords.__contains__, "BCDGLW")))

        return [attack, defense] + keywords

    @staticmethod
    def encode_players(current, opposing, version="1.2"):
        if version == "1.5":
            return (
                current.health / 30,
                current.mana / 13,
                (1 + current.bonus_draw) / 6,
                opposing.health / 30,
                (opposing.base_mana + opposing.bonus_mana) / 13,
                (1 + opposing.bonus_draw) / 6,
            )
        else:
            return (
                current.health / 30,
                current.mana / 13,
                current.next_rune / 30,
                (1 + current.bonus_draw) / 6,
                opposing.health / 30,
                (opposing.base_mana + opposing.bonus_mana) / 13,
                opposing.next_rune / 30,
                (1 + opposing.bonus_draw) / 6,
            )

    def encode_state(self):
        """Encodes a state object into a numerical matrix."""
        if self.state.phase == Phase.DECK_BUILDING:
            return self._encode_state_deck_building()
        elif self.state.phase == Phase.BATTLE:
            return self._encode_state_battle()

    @abstractmethod
    def _encode_state_deck_building(self):
        """Encodes a state object in the draft phase."""
        pass

    @abstractmethod
    def _encode_state_battle(self):
        """Encodes a state object in the battle phase."""
        pass

    @property
    def turn(self):
        return self.state.turn

    @property
    def action_mask(self):
        return self.state.action_mask

    def action_masks(self):
        """
        Method implemented especially for SB3-Contrib's MaskablePPO support.
        More at https://sb3-contrib.readthedocs.io
        """
        return self.state.action_mask

    @property
    def available_actions(self):
        return self.state.available_actions

    @property
    def _draft_is_finished(self):
        return self.state.phase > Phase.DRAFT

    @property
    def _construction_is_finished(self):
        return self.state.phase > Phase.CONSTRUCTED

    @property
    def _battle_is_finished(self):
        return self.state.phase > Phase.BATTLE
