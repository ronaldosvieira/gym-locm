from operator import attrgetter
from typing import Tuple

import numpy as np

from gym_locm.engine.player import Player
from gym_locm.engine.action import Action
from gym_locm.engine.card import (
    Card,
    Creature,
    GreenItem,
    RedItem,
    BlueItem,
)
from gym_locm.engine.enums import *
from gym_locm.engine.phases import (
    Version12BattlePhase,
    DraftPhase,
    Version15BattlePhase,
    ConstructedPhase,
)
from gym_locm.exceptions import GameIsEndedError


class State:
    def __init__(
        self,
        seed=None,
        version="1.5",
        items=True,
        deck_building_kwargs=None,
        battle_kwargs=None,
    ):
        if deck_building_kwargs is None:
            deck_building_kwargs = dict()

        if battle_kwargs is None:
            battle_kwargs = dict()

        self.rng = np.random.default_rng(seed=seed)
        self.items = items
        self.version = version
        self.was_last_action_invalid = False

        self.players = (Player(PlayerOrder.FIRST), Player(PlayerOrder.SECOND))

        if version == "1.5":
            self.deck_building_phase = ConstructedPhase(
                self, self.rng, items=items, **deck_building_kwargs
            )
            self.battle_phase = Version15BattlePhase(self, self.rng, items=items)

            self.phase = Phase.CONSTRUCTED

        elif version == "1.2":
            self.deck_building_phase = DraftPhase(
                self, self.rng, items=items, **deck_building_kwargs
            )
            self.battle_phase = Version12BattlePhase(self, self.rng, items=items)

            self.phase = Phase.DRAFT

        else:
            raise ValueError(
                f'Invalid version {version}. Supported versions: "1.5" and "1.2"'
            )

        self._phase = self.deck_building_phase
        self._phase.prepare()

    @property
    def _current_player(self) -> PlayerOrder:
        return self._phase._current_player

    @property
    def current_player(self) -> Player:
        return self.players[self._current_player]

    @property
    def opposing_player(self) -> Player:
        return self.players[(int(self._current_player) + 1) % 2]

    @property
    def available_actions(self) -> Tuple[Action]:
        return self._phase.available_actions()

    @property
    def action_mask(self):
        return self._phase.action_mask()

    @property
    def winner(self):
        return self.battle_phase.winner

    @property
    def turn(self):
        return self._phase.turn

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)

        return [seed]

    def act(self, action: Action):
        self.was_last_action_invalid = False

        self._phase.act(action)

        if self._phase.ended:
            if self.phase == Phase.DECK_BUILDING:
                self.phase = Phase.BATTLE
                self._phase = self.battle_phase
                self._phase.prepare()

            elif self.phase == Phase.BATTLE:
                self.phase = Phase.ENDED

            elif self.phase == Phase.ENDED:
                raise GameIsEndedError

    def clone(self) -> "State":
        cloned_state = State.empty_copy()

        cloned_state.rng = np.random.default_rng()

        cloned_state.rng.bit_generator.state = self.rng.bit_generator.state.copy()

        cloned_state.items = self.items
        cloned_state.version = self.version
        cloned_state.was_last_action_invalid = self.was_last_action_invalid

        cloned_state.players = tuple([player.clone() for player in self.players])

        cloned_state.phase = self.phase
        cloned_state.deck_building_phase = self.deck_building_phase.clone(cloned_state)
        cloned_state.battle_phase = self.battle_phase.clone(cloned_state)

        if self.phase == Phase.BATTLE:
            cloned_state._phase = cloned_state.battle_phase
        elif self.phase == Phase.DECK_BUILDING:
            cloned_state._phase = cloned_state.deck_building_phase

        return cloned_state

    def __str__(self) -> str:
        encoding = ""

        p, o = self.current_player, self.opposing_player

        for cp in p, o:
            draw = cp.last_drawn if cp == self.current_player else 1 + cp.bonus_draw
            draw = 0 if self.is_deck_building() else draw

            if self.version == "1.5":
                deck_length = 0 if self.is_deck_building() else len(cp.deck)

                encoding += (
                    f"{cp.health} {cp.base_mana + cp.bonus_mana} {deck_length} {draw}\n"
                )
            elif self.version == "1.2":
                encoding += f"{cp.health} {cp.base_mana + cp.bonus_mana} {len(cp.deck)} {cp.next_rune} {draw}\n"

        op_hand = len(o.hand) if self.phase != Phase.DECK_BUILDING else 0
        last_actions = []

        for action in reversed(o.actions[:-1]):
            if action.type == ActionType.PASS:
                break

            last_actions.append(action)

        encoding += f"{op_hand} {len(last_actions)}\n"

        for a in reversed(last_actions):
            target_id = -1 if a.target is None else a.target

            if isinstance(target_id, Card):
                target_id = target_id.instance_id

            encoding += (
                f"{a.resolved_origin.id} {a.type.name} " f"{a.origin} {target_id}\n"
            )

        cards = (
            p.hand
            + sorted(p.lanes[0] + p.lanes[1], key=attrgetter("summon_counter"))
            + sorted(o.lanes[0] + o.lanes[1], key=attrgetter("summon_counter"))
        )

        encoding += f"{len(cards)}\n"

        for c in cards:
            if c in p.hand:
                c.location = 0
                c.lane = -1
            elif c in p.lanes[0] + p.lanes[1]:
                c.location = 1
                c.lane = 0 if c in p.lanes[0] else 1
            elif c in o.lanes[0] + o.lanes[1]:
                c.location = -1
                c.lane = 0 if c in o.lanes[0] else 1

            if isinstance(c.type, int):
                c.cardType = c.type
            elif c.type == "creature":
                c.cardType = 0
            elif c.type == "itemGreen":
                c.cardType = 1
            elif c.type == "itemRed":
                c.cardType = 2
            elif c.type == "itemBlue":
                c.cardType = 3

            abilities = list("------")

            for i, a in enumerate(list("BCDGLW")):
                if c.has_ability(a):
                    abilities[i] = a

            c.abilities = "".join(abilities)

            c.instance_id = -1 if c.instance_id is None else c.instance_id

        if self.version == "1.5":
            for i, c in enumerate(cards):
                encoding += (
                    f"{c.id} {c.instance_id} {c.location} {c.cardType} "
                    f"{c.cost} {c.attack} {c.defense} {c.abilities} "
                    f"{c.player_hp} {c.enemy_hp} {c.card_draw} {c.area} {c.lane} \n"
                )
        elif self.version == "1.2":
            for i, c in enumerate(cards):
                encoding += (
                    f"{c.id} {c.instance_id} {c.location} {c.cardType} "
                    f"{c.cost} {c.attack} {c.defense} {c.abilities} "
                    f"{c.player_hp} {c.enemy_hp} {c.card_draw} {c.lane} \n"
                )

        return encoding

    def is_deck_building(self):
        return self.phase == Phase.DECK_BUILDING

    def is_draft(self):
        return self.phase == Phase.DRAFT

    def is_constructed(self):
        return self.phase == Phase.CONSTRUCTED

    def is_battle(self):
        return self.phase == Phase.BATTLE

    def is_ended(self):
        return self.phase == Phase.ENDED

    @staticmethod
    def empty_copy():
        class Empty(State):
            def __init__(self):
                pass

        new_copy = Empty()
        new_copy.__class__ = State

        return new_copy

    @staticmethod
    def from_native_input(game_input, deck_orders=((), ())):
        if isinstance(game_input, str):
            game_input = game_input.split("\n")

        game_input = iter(game_input)

        deck_sizes = [-1, -1]

        player_data = list(map(int, next(game_input).split()))

        if len(player_data) == 4:
            version = "1.5"

            health, mana, deck, draw = player_data
            rune = None
        else:
            version = "1.2"

            health, mana, deck, rune, draw = player_data

        state = State(version=version)
        cp, op = state.players

        cp.health = health
        cp.mana = mana
        cp.base_mana = mana
        cp.next_rune = rune
        cp.bonus_draw = 0
        cp.last_drawn = draw

        cp.hand = []
        deck_sizes[0] = deck

        player_data = list(map(int, next(game_input).split()))

        if version == "1.5":
            health, mana, deck, draw = player_data
            rune = None
        else:
            health, mana, deck, rune, draw = player_data

        op.health = health
        op.mana = mana
        op.base_mana = mana
        op.next_rune = rune
        op.bonus_draw = draw - 1
        op.last_drawn = 1

        op.hand = []
        deck_sizes[1] = deck

        if mana != 0:
            state.phase = Phase.BATTLE
            state._phase = state.battle_phase
        else:
            state._phase.turn = deck + 1

        opp_hand, opp_actions = map(int, next(game_input).split())

        # add known cards to opponent's hand
        state.opposing_player.hand = [card for _, card in deck_orders[1][:opp_hand]]

        # fill the rest of the opponent's hand with mockup cards
        state.opposing_player.hand += [
            Card.mockup_card()
            for _ in range(opp_hand - len(state.opposing_player.hand))
        ]

        # ensure we respect the current amount of cards in the opponent's hand
        state.opposing_player.hand = state.opposing_player.hand[:opp_hand]

        for _ in range(opp_actions):
            next(game_input)

        card_count = int(next(game_input))

        for _ in range(card_count):
            if version == "1.5":
                (
                    card_id,
                    instance_id,
                    location,
                    card_type,
                    cost,
                    attack,
                    defense,
                    keywords,
                    player_hp,
                    opp_hp,
                    card_draw,
                    area,
                    lane,
                ) = next(game_input).split()
            else:
                (
                    card_id,
                    instance_id,
                    location,
                    card_type,
                    cost,
                    attack,
                    defense,
                    keywords,
                    player_hp,
                    opp_hp,
                    card_draw,
                    lane,
                ) = next(game_input).split()

                area = 0

            card_type = int(card_type)

            types_dict = {0: Creature, 1: GreenItem, 2: RedItem, 3: BlueItem}

            card_class = types_dict[card_type]

            card = card_class(
                int(card_id),
                "",
                card_type,
                int(cost),
                int(attack),
                int(defense),
                keywords,
                int(player_hp),
                int(opp_hp),
                int(card_draw),
                int(area),
                "",
                instance_id=int(instance_id),
            )

            location = int(location)
            lane = int(lane)

            if location == 0:
                state.players[0].hand.append(card)
            elif location == 1:
                state.players[0].lanes[lane].append(card)
            elif location == -1:
                state.players[1].lanes[lane].append(card)

        for i, player in enumerate(state.players):
            # add known cards in the deck
            player.deck = [card for _, card in deck_orders[i]]

            # remove from the players' deck cards that are already in their hands
            for card in player.hand:
                try:
                    player.deck.remove(card)
                except ValueError:
                    pass

            # fill the rest of the deck with mockup cards
            player.deck += [
                Card.mockup_card() for _ in range(deck_sizes[i] - len(player.deck))
            ]

            # ensure we respect the correct amount of cards in the player's deck
            player.deck = player.deck[: deck_sizes[i]]

            # since we draw with player.deck.pop(), reverse the deck list
            player.deck = list(reversed(player.deck))

        if state.phase == Phase.DECK_BUILDING:
            state.opposing_player.hand = list(state.current_player.hand)

        return state


Game = State
