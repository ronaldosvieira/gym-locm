import copy
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from gym_locm.engine import (
    Card,
    Creature,
    Action,
    ActionType,
    PlayerOrder,
    get_locm12_card_list, Lane, GreenItem, RedItem, BlueItem, Location,
)
from gym_locm.exceptions import FullHandError, EmptyDeckError, NotEnoughManaError, MalformedActionError, FullLaneError, \
    InvalidCardError, WardShieldError
from gym_locm.util import is_it, has_enough_mana


class Phase(ABC):
    def __init__(self, state, rng: np.random.Generator, items=True):
        self.state = state
        self.rng = rng
        self.items = items

        self.turn = 1
        self.ended = False

        self._current_player = None
        self.__available_actions = None
        self.__action_mask = None

    @abstractmethod
    def available_actions(self) -> Tuple[Action]:
        pass

    @abstractmethod
    def action_mask(self) -> Tuple[bool]:
        pass

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def act(self, action: Action):
        pass

    @abstractmethod
    def _next_turn(self):
        pass

    def clone(self):
        return copy.deepcopy(self)


class DeckBuildingPhase(Phase, ABC):
    def __init__(self, state, rng, items=True):
        super().__init__(state, rng, items)

        # get references of the players' deck
        self.decks = state.players[0].deck, state.players[1].deck


class DraftPhase(DeckBuildingPhase):
    def __init__(self, state, rng, k=3, n=30, items=True):
        super().__init__(state, rng, items)

        self.k, self.n = k, n

        self._draft_cards = None

        self.__available_actions = self.__available_actions = tuple(
            [Action(ActionType.PICK, i) for i in range(self.k)]
        )
        self.__action_mask = tuple([True] * self.k)

    def available_actions(self) -> Tuple[Action]:
        return self.__available_actions if not self.ended else ()

    def action_mask(self) -> Tuple[bool]:
        return self.__action_mask if not self.ended else ()

    def prepare(self):
        # initialize current player pointer
        self._current_player = PlayerOrder.FIRST

        # initialize random draft cards
        self._draft_cards = self._new_draft()

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
        self.decks[self._current_player].append(card)

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


class ConstructedPhase(DeckBuildingPhase):
    def available_actions(self) -> Tuple[Action]:
        pass

    def action_mask(self) -> Tuple[bool]:
        pass

    def prepare(self):
        pass

    def act(self, action: Action):
        pass

    def _next_turn(self):
        pass


class BattlePhase(Phase, ABC):
    def __init__(self, state, rng, items=True):
        super().__init__(state, rng, items)

        self.winner = None

        self.instance_counter = 0
        self.summon_counter = 0

    def _next_instance_id(self):
        self.instance_counter += 1

        return self.instance_counter


class Version12BattlePhase(BattlePhase):
    def __init__(self, state, rng, items=True):
        super().__init__(state, rng, items)

    def available_actions(self) -> Tuple[Action]:
        if self.__available_actions is None:
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

            self.__available_actions = tuple(available_actions)

        return self.__available_actions

    def action_mask(self) -> Tuple[bool]:
        if self.__action_mask is None:
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

            self.__action_mask = action_mask

        return self.__action_mask

    def prepare(self):
        """Prepare all game components for a battle phase"""
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
            player.draw(4)
            player.base_mana = 0

        second_player = players[PlayerOrder.SECOND]
        second_player.draw()
        second_player.bonus_mana = 1

    def act(self, action: Action):
        """Execute the actions intended by the player in this battle turn"""
        origin, target = action.origin, action.target

        if isinstance(action.origin, int):
            origin = self._find_card(origin)

        if action.type == ActionType.SUMMON:
            if isinstance(action.target, int):
                target = Lane(target)

            self._do_summon(origin, target)
        elif action.type == ActionType.ATTACK:
            if isinstance(action.target, int):
                target = self._find_card(target)

            self._do_attack(origin, target)
        elif action.type == ActionType.USE:
            if isinstance(action.target, int):
                target = self._find_card(target)

            self._do_use(origin, target)
        elif action.type == ActionType.PASS:
            self._next_turn()
        else:
            raise MalformedActionError("Invalid action type")

        action.resolved_origin = origin
        action.resolved_target = target

        self.state.current_player.actions.append(action)

        players = self.state.players

        for player in players:
            for lane in player.lanes:
                for creature in lane:
                    if creature.is_dead:
                        lane.remove(creature)

        if players[PlayerOrder.FIRST].health <= 0:
            self.ended = True
            self.winner = PlayerOrder.SECOND
        elif players[PlayerOrder.SECOND].health <= 0:
            self.ended = True
            self.winner = PlayerOrder.FIRST

    def _find_card(self, instance_id: int) -> Card:
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
                    return card

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
        current_player.damage(-origin.player_hp)
        opposing_player.damage(-origin.enemy_hp)

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
            damage_dealt = opposing_player.damage(origin.attack)

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
                opposing_player.damage(excess_damage)
        else:
            raise MalformedActionError("Target is not a creature or a player")

        if "D" in origin.keywords:
            current_player.health += damage_dealt

        origin.has_attacked_this_turn = True

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

        if isinstance(origin, GreenItem):
            is_own_creature = (
                    target in current_player.lanes[Lane.LEFT]
                    or target in current_player.lanes[Lane.RIGHT]
            )

            if target is None or not is_own_creature:
                error = "Green items should be used on friendly creatures"
                raise MalformedActionError(error)

            target.attack = max(0, target.attack + origin.attack)
            target.defense += origin.defense
            target.keywords = target.keywords.union(origin.keywords)

            if target.defense <= 0:
                target.is_dead = True

            current_player.bonus_draw += origin.card_draw
            current_player.damage(-origin.player_hp)
            opposing_player.damage(-origin.enemy_hp)

        elif isinstance(origin, RedItem):
            is_opp_creature = (
                    target in opposing_player.lanes[Lane.LEFT]
                    or target in opposing_player.lanes[Lane.RIGHT]
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

            current_player.bonus_draw += origin.card_draw
            current_player.damage(-origin.player_hp)
            opposing_player.damage(-origin.enemy_hp)

        elif isinstance(origin, BlueItem):
            is_opp_creature = (
                    target in opposing_player.lanes[Lane.LEFT]
                    or target in opposing_player.lanes[Lane.RIGHT]
            )

            if target is not None and not is_opp_creature:
                error = (
                    "Blue items should be used on enemy creatures or enemy player"
                )
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
                opposing_player.damage(-origin.defense)
            else:
                raise MalformedActionError("Invalid target")

            current_player.bonus_draw += origin.card_draw
            current_player.damage(-origin.player_hp)
            opposing_player.damage(-origin.enemy_hp)

        else:
            error = "Card being used is not an item"
            raise MalformedActionError(error)

        current_player.hand.remove(origin)
        current_player.mana -= origin.cost

    def _next_turn(self):
        # invalidate cached action list and masks
        self.__available_actions = None
        self.__action_mask = None

        # handle turn change
        if self._current_player == PlayerOrder.FIRST:
            self._current_player = PlayerOrder.SECOND
        else:
            self._current_player = PlayerOrder.FIRST

            self.turn += 1

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
            current_player.deck = []

        try:
            current_player.draw(amount_to_draw)
        except FullHandError:
            pass
        except EmptyDeckError as e:
            for _ in range(e.remaining_draws):
                deck_burn = current_player.health - current_player.next_rune
                current_player.damage(deck_burn)

        current_player.bonus_draw = 0
        current_player.last_drawn = amount_to_draw
