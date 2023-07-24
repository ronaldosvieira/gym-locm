import copy
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from gym_locm.engine.card_generator import generate_card
from gym_locm.engine.action import Action
from gym_locm.engine.player import Player
from gym_locm.engine.card import (
    get_locm12_card_list,
    Creature,
    Card,
    GreenItem,
    RedItem,
    BlueItem,
)
from gym_locm.engine.enums import (
    DamageSource,
    ActionType,
    PlayerOrder,
    Lane,
    Area,
    Location,
)
from gym_locm.exceptions import (
    FullHandError,
    EmptyDeckError,
    NotEnoughManaError,
    MalformedActionError,
    FullLaneError,
    InvalidCardError,
    WardShieldError,
)
from gym_locm.util import is_it, has_enough_mana


class Phase(ABC):
    def __init__(self, state, rng: np.random.Generator, *, items=True):
        self.state = state
        self.rng = rng
        self.items = items

        self.turn = 1
        self.prepared = False
        self.ended = False

        self._current_player = PlayerOrder.FIRST
        self._available_actions = None
        self._action_mask = None

    @abstractmethod
    def available_actions(self) -> Tuple[Action]:
        pass

    @abstractmethod
    def action_mask(self) -> Tuple[bool]:
        pass

    @abstractmethod
    def prepare(self):
        self.prepared = True

    @abstractmethod
    def act(self, action: Action):
        if not self.prepared:
            raise Exception("Must call prepare() before act()")

    @abstractmethod
    def _next_turn(self):
        pass

    def clone(self, cloned_state):
        cloned_phase = self.empty_copy(type(self))

        cloned_phase.state = cloned_state
        cloned_phase.rng = cloned_state.rng

        cloned_phase.items = self.items
        cloned_phase.turn = self.turn
        cloned_phase.ended = self.ended

        cloned_phase._current_player = self._current_player
        cloned_phase._available_actions = self._available_actions
        cloned_phase._action_mask = self._action_mask

        return cloned_phase

    @staticmethod
    def empty_copy(of_class):
        class Empty(of_class):
            def __init__(self):
                pass

        new_copy = Empty()
        new_copy.__class__ = of_class

        return new_copy


class DeckBuildingPhase(Phase, ABC):
    def __init__(self, state, rng, *, items=True):
        super().__init__(state, rng, items=items)


class DraftPhase(DeckBuildingPhase):
    def __init__(self, state, rng, *, items=True, k=3, n=30):
        super().__init__(state, rng, items=items)

        self.k, self.n = k, n

        self._draft_cards = None

        self._available_actions = self._available_actions = tuple(
            [Action(ActionType.PICK, i) for i in range(self.k)]
        )
        self._action_mask = tuple([True] * self.k)

    def available_actions(self) -> Tuple[Action]:
        return self._available_actions if not self.ended else ()

    def action_mask(self) -> Tuple[bool]:
        return self._action_mask if not self.ended else ()

    def prepare(self):
        super().prepare()

        # initialize current player pointer
        self._current_player = PlayerOrder.FIRST

        # initialize random draft cards
        self._draft_cards = self._new_draft()

        # initialize the players' hands
        for player in self.state.players:
            player.hand = self.current_choices

    def _new_draft(self):
        # retrieve card list
        cards = list(get_locm12_card_list())

        # if items are not wanted, filter them out of the card list
        if not self.items:
            cards = list(filter(is_it(Creature), cards))

        # get 60 random cards from the card list
        self.rng.shuffle(cards)
        pool = cards[:60]

        # get 3 random cards without replacement for each turn
        draft = []

        for _ in range(self.n):
            self.rng.shuffle(pool)

            draft.append(pool[: self.k])

        return draft

    def act(self, action: Action):
        """Execute the action intended by the player in this draft turn"""
        # get chosen card
        chosen_index = action.origin if action.origin is not None else 0
        card = self.current_choices[chosen_index]

        # add chosen card to player's deck
        self.state.players[self._current_player].deck.append(card)

        # trigger next turn
        self._next_turn()

    def _next_turn(self):
        # handle turn change
        if self._current_player == PlayerOrder.FIRST:
            self._current_player = PlayerOrder.SECOND
        else:
            if self.turn < self.n:
                self._current_player = PlayerOrder.FIRST

                self.turn += 1

                for player in self.state.players:
                    player.hand = self.current_choices
            else:
                self._current_player = None
                self.ended = True

    @property
    def current_choices(self) -> List[Card]:
        try:
            return self._draft_cards[self.turn - 1]
        except IndexError:
            return []

    def clone(self, cloned_state):
        cloned_phase = super().clone(cloned_state)

        cloned_phase.k = self.k
        cloned_phase.n = self.n
        cloned_phase._draft_cards = self._draft_cards
        cloned_phase._available_actions = self._available_actions
        cloned_phase._action_mask = self._action_mask

        return cloned_phase


class ConstructedPhase(DeckBuildingPhase):
    def __init__(self, state, rng, *, items=True, k=120, n=30, max_copies=2):
        super().__init__(state, rng, items=items)

        self.k, self.n = k, n
        self.max_copies = max_copies

        self._constructed_cards = None
        self._action_mask = [True] * self.k, [True] * self.k
        self._choices = [0] * self.k, [0] * self.k

    def available_actions(self) -> Tuple[Action]:
        return tuple(
            Action(ActionType.CHOOSE, i)
            for i, can_be_chosen in enumerate(self.action_mask())
            if can_be_chosen
        )

    def action_mask(self) -> Tuple[bool]:
        return tuple(self._action_mask[self._current_player])

    def prepare(self):
        super().prepare()

        # initialize current player pointer
        self._current_player = PlayerOrder.FIRST

        # initialize random constructed cards
        self._constructed_cards = self._new_constructed()

        # initialize the players' hands
        for player in self.state.players:
            player.hand = list(self._constructed_cards)

    def _new_constructed(self):
        card_pool = [generate_card(i, self.rng, self.items) for i in range(self.k)]

        return card_pool

    def act(self, action: Action):
        # get chosen card
        if action.type == ActionType.CHOOSE:
            chosen_card_id = action.origin
        elif action.type == ActionType.PASS:
            chosen_card_id = self.action_mask().index(
                True
            )  # get first choose-able card
        else:
            raise MalformedActionError(
                f"Actions in constructed should be of types CHOOSE or PASS, not {action.type}"
            )

        chosen_card_index = chosen_card_id

        # validate choice
        if 0 >= chosen_card_index >= self.k:
            raise MalformedActionError(f"Invalid card ID: {chosen_card_id}")

        player_choices = self._choices[self._current_player]

        if not self.action_mask()[chosen_card_index]:
            raise MalformedActionError(
                f"Can't choose more copies of card {chosen_card_id}"
            )

        # increase times chosen
        player_choices[chosen_card_index] += 1

        # update action mask, if needed
        if player_choices[chosen_card_index] >= self.max_copies:
            self._action_mask[self._current_player][chosen_card_index] = False

        # add chosen card to player's deck
        card = self._constructed_cards[chosen_card_index]
        self.state.players[self._current_player].deck.append(card)

        # trigger next turn
        self._next_turn()

    def _next_turn(self):
        # handle turn change
        if self._current_player == PlayerOrder.FIRST:
            if self.turn < self.n:
                self.turn += 1
            else:
                self._current_player = PlayerOrder.SECOND

                self.turn = 1
        else:
            if self.turn < self.n:
                self.turn += 1
            else:
                self._current_player = None
                self.ended = True

        # populate players' hands
        if not self.ended:
            for player in self.state.players:
                player.hand = self._constructed_cards
        else:
            for player in self.state.players:
                player.hand = []

    def clone(self, cloned_state):
        cloned_phase = super().clone(cloned_state)

        cloned_phase.k = self.k
        cloned_phase.n = self.n
        cloned_phase.max_copies = self.max_copies
        cloned_phase._constructed_cards = self._constructed_cards
        cloned_phase._action_mask = list(self._action_mask[0]), list(
            self._action_mask[1]
        )
        cloned_phase._choices = list(self._choices[0]), list(self._choices[1])

        return cloned_phase


class BattlePhase(Phase):
    def __init__(self, state, rng, *, items=True):
        super().__init__(state, rng, items=items)

        self.winner = None

        self.instance_counter = 0
        self.summon_counter = 0
        self.damage_counter = [0, 0]

    def _next_instance_id(self):
        self.instance_counter += 1

        return self.instance_counter

    def available_actions(self) -> Tuple[Action]:
        if self._available_actions is None:
            current_player = self.state.players[self._current_player]
            opposing_player = self.state.players[self._current_player.opposing()]

            summon, attack, use = [], [], []

            c_hand = current_player.hand
            c_lanes = current_player.lanes
            o_lanes = opposing_player.lanes

            for card in filter(has_enough_mana(current_player.mana), c_hand):
                origin = card.instance_id

                if isinstance(card, Creature):
                    for lane in Lane:
                        if len(c_lanes[lane]) < 3:
                            summon.append(Action(ActionType.SUMMON, origin, lane))

                elif isinstance(card, GreenItem):
                    for lane in Lane:
                        for friendly_creature in c_lanes[lane]:
                            target = friendly_creature.instance_id

                            use.append(Action(ActionType.USE, origin, target))

                elif isinstance(card, RedItem):
                    for lane in Lane:
                        for enemy_creature in o_lanes[lane]:
                            target = enemy_creature.instance_id

                            use.append(Action(ActionType.USE, origin, target))

                elif isinstance(card, BlueItem):
                    for lane in Lane:
                        for enemy_creature in o_lanes[lane]:
                            target = enemy_creature.instance_id

                            use.append(Action(ActionType.USE, origin, target))

                    use.append(Action(ActionType.USE, origin, None))

            for lane in Lane:
                guard_creatures = []

                for enemy_creature in o_lanes[lane]:
                    if enemy_creature.has_ability("G"):
                        guard_creatures.append(enemy_creature)

                if not guard_creatures:
                    valid_targets = o_lanes[lane] + [None]
                else:
                    valid_targets = guard_creatures

                for friendly_creature in filter(Creature.able_to_attack, c_lanes[lane]):
                    origin = friendly_creature.instance_id

                    for valid_target in valid_targets:
                        if valid_target is not None:
                            valid_target = valid_target.instance_id

                        attack.append(Action(ActionType.ATTACK, origin, valid_target))

            available_actions = [Action(ActionType.PASS)] + summon + use + attack

            self._available_actions = tuple(available_actions)

        return self._available_actions

    def action_mask(self) -> Tuple[bool]:
        if self._action_mask is None:
            action_mask = [False] * 145

            # pass is always allowed
            action_mask[0] = True

            # shortcuts
            cp = self.state.players[self._current_player]
            op = self.state.players[self._current_player.opposing()]
            cp_has_enough_mana = has_enough_mana(cp.mana)
            left_lane_not_full = len(cp.lanes[0]) < 3
            right_lane_not_full = len(cp.lanes[1]) < 3

            def validate_creature(index):
                if left_lane_not_full:
                    action_mask[1 + index * 2] = True

                if right_lane_not_full:
                    action_mask[1 + index * 2 + 1] = True

            def validate_green_item(index):
                for i in range(len(cp.lanes[0])):
                    action_mask[17 + index * 13 + 1 + i] = True

                for i in range(len(cp.lanes[1])):
                    action_mask[17 + index * 13 + 4 + i] = True

            def validate_red_item(index):
                for i in range(len(op.lanes[0])):
                    action_mask[17 + index * 13 + 7 + i] = True

                for i in range(len(op.lanes[1])):
                    action_mask[17 + index * 13 + 10 + i] = True

            def validate_blue_item(index):
                validate_red_item(index)

                action_mask[17 + index * 13] = True

            check_playability = {
                Creature: validate_creature,
                GreenItem: validate_green_item,
                RedItem: validate_red_item,
                BlueItem: validate_blue_item,
            }

            # for each card in hand, check valid actions
            for i, card in enumerate(cp.hand):
                if cp_has_enough_mana(card):
                    check_playability[type(card)](i)

            # for each card in the board, check valid actions
            for offset, lane_id in zip((0, 3), (0, 1)):
                for i, creature in enumerate(cp.lanes[lane_id]):
                    i += offset

                    if creature.able_to_attack():
                        guards = []

                        for j, enemy_creature in enumerate(op.lanes[lane_id]):
                            if enemy_creature.has_ability("G"):
                                guards.append(j)

                        if guards:
                            for j in guards:
                                action_mask[121 + i * 4 + 1 + j] = True
                        else:
                            action_mask[121 + i * 4] = True

                            for j in range(len(op.lanes[lane_id])):
                                action_mask[121 + i * 4 + 1 + j] = True

            if not self.items:
                action_mask = action_mask[:17] + action_mask[-24:]

            self._action_mask = action_mask

        return self._action_mask

    def prepare(self):
        super().prepare()

        """Prepare all game components for a battle phase"""
        self._current_player = PlayerOrder.FIRST

        players = self.state.players

        for player in players:
            player.hand = []
            player.lanes = ([], [])

            self.rng.shuffle(player.deck)

        d1, d2 = [], []

        for card1, card2 in zip(*(p.deck for p in players)):
            d1.append(card1.make_copy(self._next_instance_id()))
            d2.append(card2.make_copy(self._next_instance_id()))

        players[0].deck = list(reversed(d1))
        players[1].deck = list(reversed(d2))

        for player in players:
            self._draw(4, player=player)
            player.base_mana = 0

        second_player = players[PlayerOrder.SECOND]
        self._draw(player=second_player)
        second_player.bonus_mana = 1

        self._new_battle_turn()

    def act(self, action: Action):
        """Execute the actions intended by the player in this battle turn"""
        origin, target = action.origin, action.target

        if isinstance(action.origin, int):
            origin, _ = self._find_card(origin)

        if action.type == ActionType.SUMMON:
            if isinstance(action.target, int):
                target = Lane(target)

            self._do_summon(origin, target)
        elif action.type == ActionType.ATTACK:
            if isinstance(action.target, int):
                target, _ = self._find_card(target)

            self._do_attack(origin, target)
        elif action.type == ActionType.USE:
            if isinstance(action.target, int):
                target, _ = self._find_card(target)

            self._do_use(origin, target)
        elif action.type == ActionType.PASS:
            pass
        else:
            raise MalformedActionError("Invalid action type")

        action.resolved_origin = origin
        action.resolved_target = target

        self.state.current_player.actions.append(action)

        players = self.state.players

        for player in players:
            for lane in player.lanes:
                creatures_to_remove = []

                for creature in lane:
                    if creature.is_dead:
                        creatures_to_remove.append(creature)

                for creature in creatures_to_remove:
                    lane.remove(creature)

        if action.type == ActionType.PASS:
            self._next_turn()

        self._check_win_conditions()

        # invalidate cached action list and masks
        self._available_actions = None
        self._action_mask = None

    def _check_win_conditions(self):
        if self.state.players[PlayerOrder.FIRST].health <= 0:
            self.ended = True
            self.winner = PlayerOrder.SECOND
        elif self.state.players[PlayerOrder.SECOND].health <= 0:
            self.ended = True
            self.winner = PlayerOrder.FIRST

    def _damage_player(self, player: Player, amount: int, source: DamageSource) -> int:
        player.health -= amount

        if amount > 0 and source == DamageSource.OPPONENT:
            self.damage_counter[player.id] += amount

            while self.damage_counter[player.id] >= 5:
                self.damage_counter[player.id] -= 5
                player.bonus_draw += 1

        return amount

    def _draw(self, amount: int = 1, player: Player = None):
        if player is None:
            player = self.state.current_player

        for i in range(amount):
            if len(player.hand) >= 8:
                raise FullHandError()

            if len(player.deck) == 0:
                raise EmptyDeckError(amount - i)

            player.hand.append(player.deck.pop())

    def _handle_draw_from_empty_deck(self, remaining_draws: int = 1):
        self._damage_player(
            self.state.current_player,
            amount=10 * remaining_draws,
            source=DamageSource.GAME,
        )

    def _handle_turn_51_or_greater(self):
        self._damage_player(
            self.state.current_player, amount=10, source=DamageSource.GAME
        )

    def _find_card(self, instance_id: int) -> Tuple[Card, Location]:
        # todo: use an instance_id -> card mapping like in the original engine
        c = self.state.players[self._current_player]
        o = self.state.players[self._current_player.opposing()]

        location_mapping = {
            Location.PLAYER_HAND: c.hand,
            Location.ENEMY_HAND: o.hand,
            Location.PLAYER_LEFT_LANE: c.lanes[0],
            Location.PLAYER_RIGHT_LANE: c.lanes[1],
            Location.ENEMY_LEFT_LANE: o.lanes[0],
            Location.ENEMY_RIGHT_LANE: o.lanes[1],
        }

        for location, cards in location_mapping.items():
            for card in cards:
                if card.instance_id == instance_id:
                    return card, location

        raise InvalidCardError(instance_id)

    def _do_summon(self, origin, target):
        current_player = self.state.players[self._current_player]
        opposing_player = self.state.players[self._current_player.opposing()]

        if origin.cost > current_player.mana:
            raise NotEnoughManaError()

        if not isinstance(origin, Creature):
            raise MalformedActionError("Card being summoned is not a creature")

        if not isinstance(target, Lane):
            raise MalformedActionError("Target is not a lane")

        if len(current_player.lanes[target]) >= 3:
            raise FullLaneError()

        try:
            current_player.hand.remove(origin)
        except ValueError:
            raise MalformedActionError("Card is not in player's hand")

        origin.can_attack = False
        origin.summon_counter = self.summon_counter

        self.summon_counter += 1

        current_player.lanes[target].append(origin)

        current_player.bonus_draw += origin.card_draw
        self._damage_player(
            current_player, amount=-origin.player_hp, source=DamageSource.SELF
        )
        self._damage_player(
            opposing_player, amount=-origin.enemy_hp, source=DamageSource.OPPONENT
        )

        if origin.area != Area.NONE:
            if origin.area == Area.TYPE_1:
                target_copy = target
            elif origin.area == Area.TYPE_2:
                target_copy = target.opposing()
            else:
                raise InvalidCardError(message=f"Invalid area value: {origin.area}")

            if len(current_player.lanes[target_copy]) < 3:
                origin_copy = origin.make_copy(self._next_instance_id())

                origin_copy.summon_counter = self.summon_counter
                self.summon_counter += 1

                current_player.lanes[target_copy].append(origin_copy)

                current_player.bonus_draw += origin_copy.card_draw
                self._damage_player(
                    current_player,
                    amount=-origin_copy.player_hp,
                    source=DamageSource.SELF,
                )
                self._damage_player(
                    opposing_player,
                    amount=-origin_copy.enemy_hp,
                    source=DamageSource.OPPONENT,
                )

        current_player.mana -= origin.cost

    def _do_attack(self, origin, target):
        current_player = self.state.players[self._current_player]
        opposing_player = self.state.players[self._current_player.opposing()]

        if not isinstance(origin, Creature):
            raise MalformedActionError("Attacking card is not a creature")

        if origin in current_player.lanes[Lane.LEFT]:
            origin_lane = Lane.LEFT
        elif origin in current_player.lanes[Lane.RIGHT]:
            origin_lane = Lane.RIGHT
        else:
            raise MalformedActionError("Attacking creature is not owned by player")

        guard_creatures = []

        for creature in opposing_player.lanes[origin_lane]:
            if creature.has_ability("G"):
                guard_creatures.append(creature)

        if len(guard_creatures) > 0:
            valid_targets = guard_creatures
        else:
            valid_targets = [None] + opposing_player.lanes[origin_lane]

        if target not in valid_targets:
            raise MalformedActionError("Invalid target")

        if not origin.able_to_attack():
            raise MalformedActionError("Attacking creature cannot attack")

        if target is None:
            damage_dealt = self._damage_player(
                opposing_player, amount=origin.attack, source=DamageSource.OPPONENT
            )

        elif isinstance(target, Creature):
            target_defense = target.defense

            try:
                damage_dealt = target.damage(
                    origin.attack, lethal=origin.has_ability("L")
                )
            except WardShieldError:
                damage_dealt = 0

            try:
                origin.damage(target.attack, lethal=target.has_ability("L"))
            except WardShieldError:
                pass

            excess_damage = damage_dealt - target_defense

            if "B" in origin.keywords and excess_damage > 0:
                self._damage_player(
                    opposing_player, amount=excess_damage, source=DamageSource.OPPONENT
                )
        else:
            raise MalformedActionError("Target is not a creature or a player")

        if "D" in origin.keywords:
            current_player.health += damage_dealt

        origin.has_attacked_this_turn = True

    def _do_use_green(self, origin, target):
        is_own_creature = (
            target in self.state.current_player.lanes[Lane.LEFT]
            or target in self.state.current_player.lanes[Lane.RIGHT]
        )

        if target is None or not is_own_creature:
            error = "Green items should be used on friendly creatures"
            raise MalformedActionError(error)

        target.attack = max(0, target.attack + origin.attack)
        target.defense += origin.defense
        target.keywords = target.keywords.union(origin.keywords)

        if target.defense <= 0:
            target.is_dead = True

    def _do_use_red(self, origin, target):
        is_opp_creature = (
            target in self.state.opposing_player.lanes[Lane.LEFT]
            or target in self.state.opposing_player.lanes[Lane.RIGHT]
        )

        if target is None or not is_opp_creature:
            error = "Red items should be used on enemy creatures"
            raise MalformedActionError(error)

        target.attack = max(0, target.attack + origin.attack)
        target.keywords = target.keywords.difference(origin.keywords)

        try:
            target.damage(-origin.defense)
        except WardShieldError:
            pass

        if target.defense <= 0:
            target.is_dead = True

    def _do_use_blue(self, origin, target):
        is_opp_creature = (
            target in self.state.opposing_player.lanes[Lane.LEFT]
            or target in self.state.opposing_player.lanes[Lane.RIGHT]
        )

        if target is not None and not is_opp_creature:
            error = "Blue items should be used on enemy creatures or enemy player"
            raise MalformedActionError(error)

        if isinstance(target, Creature):
            target.attack = max(0, target.attack + origin.attack)
            target.keywords = target.keywords.difference(origin.keywords)

            try:
                target.damage(-origin.defense)
            except WardShieldError:
                pass

            if target.defense <= 0:
                target.is_dead = True

        elif target is None:
            self._damage_player(
                self.state.opposing_player,
                amount=-origin.defense,
                source=DamageSource.OPPONENT,
            )
        else:
            raise MalformedActionError("Invalid target")

    def _do_use(self, origin, target):
        current_player = self.state.players[self._current_player]
        opposing_player = self.state.players[self._current_player.opposing()]

        if origin.cost > current_player.mana:
            raise NotEnoughManaError()

        if target is not None and not isinstance(target, Creature):
            error = "Target is not a creature or a player"
            raise MalformedActionError(error)

        if origin not in current_player.hand:
            raise MalformedActionError("Card is not in player's hand")

        if origin.area != Area.NONE and target is not None:
            targets = []

            _, location = self._find_card(target.instance_id)

            # note: see the Location, PlayerOrder and Lane classes for more details on this arithmetic
            opponent_is_owner = bool((location // 10) - 1)
            target_lane = Lane(location % 10)

            owner = opposing_player if opponent_is_owner else current_player

            targets.extend(owner.lanes[target_lane])

            if origin.area == Area.TYPE_2:
                targets.extend(owner.lanes[target_lane.opposing()])
        else:
            targets = [target]

        for target in targets:
            if isinstance(origin, GreenItem):
                self._do_use_green(origin, target)
            elif isinstance(origin, RedItem):
                self._do_use_red(origin, target)
            elif isinstance(origin, BlueItem):
                self._do_use_blue(origin, target)
            else:
                raise MalformedActionError("Card being used is not an item")

            current_player.bonus_draw += origin.card_draw
            self._damage_player(
                current_player, amount=-origin.player_hp, source=DamageSource.SELF
            )
            self._damage_player(
                opposing_player, amount=-origin.enemy_hp, source=DamageSource.OPPONENT
            )

        current_player.hand.remove(origin)
        current_player.mana -= origin.cost

    def _next_turn(self):
        # handle turn change
        if self._current_player == PlayerOrder.FIRST:
            self._current_player = PlayerOrder.SECOND
        else:
            self._current_player = PlayerOrder.FIRST

            self.turn += 1

        self._new_battle_turn()

    def _new_battle_turn(self):
        # reset damage counters
        self.damage_counter = [0, 0]

        current_player = self.state.players[self._current_player]

        for creature in current_player.lanes[Lane.LEFT]:
            creature.can_attack = True
            creature.has_attacked_this_turn = False

        for creature in current_player.lanes[Lane.RIGHT]:
            creature.can_attack = True
            creature.has_attacked_this_turn = False

        if current_player.base_mana > 0 and current_player.mana == 0:
            current_player.bonus_mana = 0

        if current_player.base_mana < 12:
            current_player.base_mana += 1

        current_player.mana = current_player.base_mana + current_player.bonus_mana

        amount_to_draw = 1 + current_player.bonus_draw

        if self.turn > 50:
            self._handle_turn_51_or_greater()

        try:
            self._draw(amount_to_draw, player=current_player)
        except FullHandError:
            pass
        except EmptyDeckError as e:
            self._handle_draw_from_empty_deck(e.remaining_draws)

        current_player.bonus_draw = 0
        current_player.last_drawn = amount_to_draw

    def clone(self, cloned_state):
        cloned_phase = super().clone(cloned_state)

        cloned_phase.winner = self.winner
        cloned_phase.instance_counter = self.instance_counter
        cloned_phase.summon_counter = self.summon_counter
        cloned_phase.damage_counter = list(self.damage_counter)

        return cloned_phase


Version15BattlePhase = BattlePhase


class Version12BattlePhase(BattlePhase):
    def __init__(self, state, rng, *, items=True):
        super().__init__(state, rng, items=items)

    def _damage_player(self, player: Player, amount: int, source: DamageSource) -> int:
        player.health -= amount

        while player.health <= player.next_rune and player.next_rune > 0:
            player.next_rune -= 5
            player.bonus_draw += 1

        return amount

    def _draw(self, amount: int = 1, player: Player = None):
        if player is None:
            player = self.state.current_player

        for i in range(amount):
            if len(player.deck) == 0:
                raise EmptyDeckError(amount - i)

            if len(player.hand) >= 8:
                raise FullHandError()

            player.hand.append(player.deck.pop())

    def _do_use(self, origin, target):
        super()._do_use(origin, target)

        # see: https://github.com/acatai/Strategy-Card-Game-AI-Competition/issues/7
        if isinstance(target, Creature):
            target.player_hp = 0
            target.enemy_hp = 0
            target.card_draw = 0
            target.area = 0

    def _do_attack(self, origin, target):
        super()._do_attack(origin, target)

        # see: https://github.com/acatai/Strategy-Card-Game-AI-Competition/issues/7
        origin.player_hp = 0
        origin.enemy_hp = 0
        origin.card_draw = 0
        origin.area = 0

        if isinstance(target, Creature):
            target.player_hp = 0
            target.enemy_hp = 0
            target.card_draw = 0
            target.area = 0

    def _handle_draw_from_empty_deck(self, remaining_draws: int = 1):
        cp = self.state.current_player

        for _ in range(remaining_draws):
            deck_burn = cp.health - cp.next_rune
            self._damage_player(cp, amount=deck_burn, source=DamageSource.GAME)

    def _handle_turn_51_or_greater(self):
        self.state.current_player.deck = []

    def clone(self, cloned_state):
        cloned_phase = super().clone(cloned_state)

        cloned_phase.__class__ = Version15BattlePhase

        return cloned_phase
