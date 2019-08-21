import copy
import sys
import numpy as np

from typing import List
from enum import Enum, IntEnum
from gym_locm.exceptions import *
from gym_locm.helpers import *

instance_counter = -1


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _next_instance_id():
    global instance_counter

    instance_counter += 1

    return instance_counter


class Phase(IntEnum):
    DRAFT = 0
    BATTLE = 1
    ENDED = 2


class PlayerOrder(IntEnum):
    FIRST = 0
    SECOND = 1

    def opposing(self):
        return PlayerOrder((self + 1) % 2)


class Lane(IntEnum):
    LEFT = 0
    RIGHT = 1


class ActionType(Enum):
    PICK = 0
    SUMMON = 1
    ATTACK = 2
    USE = 3
    PASS = 4


class Player:
    def __init__(self):
        self.health = 30
        self.base_mana = 0
        self.bonus_mana = 0
        self.mana = 0
        self.next_rune = 25
        self.bonus_draw = 0

        self.deck = []
        self.hand = []
        self.lanes = ([], [])

        self.actions = []

    def draw(self, amount=1):
        for _ in range(amount):
            if len(self.deck) == 0:
                raise EmptyDeckError()

            if len(self.hand) >= 8:
                raise FullHandError()

            self.hand.append(self.deck.pop())

    def damage(self, amount) -> int:
        self.health -= amount

        while self.health <= self.next_rune:
            self.next_rune -= 5
            self.bonus_draw += 1

        return amount


class Card:
    def __init__(self, card_id, name, card_type, cost, attack, defense, keywords,
                 player_hp, enemy_hp, card_draw, text):
        self.id = card_id
        self.instance_id = None
        self.name = name
        self.type = card_type
        self.cost = cost
        self.attack = attack
        self.defense = defense
        self.keywords = set(list(keywords.replace("-", "")))
        self.player_hp = player_hp
        self.enemy_hp = enemy_hp
        self.card_draw = card_draw
        self.text = text

    def has_ability(self, keyword):
        return keyword in self.keywords

    def make_copy(self):
        card = copy.copy(self)

        card.instance_id = _next_instance_id()

        return card

    def __eq__(self, other):
        return other is not None \
               and self.instance_id is not None \
               and other.instance_id is not None \
               and self.instance_id == other.instance_id

    def __repr__(self):
        return f"({self.instance_id}: {self.name})"


class Creature(Card):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_dead = False
        self.can_attack = False
        self.has_attacked_this_turn = False

    def remove_ability(self, ability):
        self.keywords.discard(ability)

    def add_ability(self, ability):
        self.keywords.add(ability)

    def able_to_attack(self):
        return not self.has_attacked_this_turn and \
               (self.can_attack or self.has_ability('C'))

    def damage(self, amount=1, lethal=False) -> int:
        if amount <= 0:
            return 0

        if self.has_ability('W'):
            self.remove_ability('W')

            raise WardShieldError()

        self.defense -= amount

        if lethal or self.defense <= 0:
            self.is_dead = True

        return amount


class Item(Card):
    pass


class GreenItem(Item):
    pass


class RedItem(Item):
    pass


class BlueItem(Item):
    pass


class GameState:
    def __init__(self, current_phase, current_player, turn, players):
        self.current_phase = current_phase
        self._current_player = current_player
        self.turn = turn
        self.players = players
        self.__available_actions = None

    @property
    def current_player(self):
        return self.players[self._current_player]

    @property
    def opposing_player(self):
        return self.players[self._current_player.opposing()]

    @property
    def available_actions(self):
        if self.__available_actions is not None:
            return self.__available_actions

        if self.current_phase == Phase.DRAFT:
            self.__available_actions = [
                Action(ActionType.PICK, 0),
                Action(ActionType.PICK, 1),
                Action(ActionType.PICK, 2)
            ]
        elif self.current_phase == Phase.ENDED:
            self.__available_actions = []
        else:
            summon, attack, use = [], [], []

            for card in filter(has_enough_mana(self.current_player.mana),
                               self.current_player.hand):
                if isinstance(card, Creature):
                    for lane in Lane:
                        if len(self.current_player.lanes[lane]) < 3:
                            summon.append(Action(ActionType.SUMMON, card, lane))
                elif isinstance(card, GreenItem):
                    for lane in Lane:
                        for friendly_creature in self.current_player.lanes[lane]:
                            use.append(Action(ActionType.USE, card, friendly_creature))
                elif isinstance(card, RedItem):
                    for lane in Lane:
                        for enemy_creature in self.opposing_player.lanes[lane]:
                            use.append(Action(ActionType.USE, card, enemy_creature))
                elif isinstance(card, BlueItem):
                    for lane in Lane:
                        for friendly_creature in self.current_player.lanes[lane]:
                            use.append(Action(ActionType.USE, card, friendly_creature))

                        for enemy_creature in self.opposing_player.lanes[lane]:
                            use.append(Action(ActionType.USE, card, enemy_creature))

                        use.append(Action(ActionType.USE, card, None))

            for lane in Lane:
                guard_creatures = []

                for enemy_creature in self.opposing_player.lanes[lane]:
                    if enemy_creature.has_ability('G'):
                        guard_creatures.append(enemy_creature)

                if not guard_creatures:
                    valid_targets = self.opposing_player.lanes[lane] + [None]
                else:
                    valid_targets = guard_creatures

                for friendly_creature in filter(Creature.able_to_attack,
                                                self.current_player.lanes[lane]):
                    for valid_target in valid_targets:
                        attack.append(Action(ActionType.ATTACK, friendly_creature, valid_target))

            available_actions = summon + attack + use

            if not available_actions:
                available_actions = [Action(ActionType.PASS)]

            self.__available_actions = available_actions

        return self.__available_actions


class Action:
    def __init__(self, action_type, origin=None, target=None):
        self.type = action_type
        self.origin = origin
        self.target = target

    def __eq__(self, other):
        return other is not None and \
               self.type == other.type and \
               self.origin == other.origin and \
               self.target == other.target


class Game:
    _draft_cards: List[List[Card]]
    current_player: PlayerOrder
    current_phase: Phase

    def __init__(self, cards_in_deck=30):
        self.cards_in_deck = cards_in_deck

        self._cards = self._load_cards()
        self.players = ()
        self.turn = -1

        self.reset()

    def reset(self) -> GameState:
        self.current_phase = Phase.DRAFT
        self.current_player = PlayerOrder.FIRST
        self.turn = 1

        self._prepare_for_draft()

        return self._build_game_state()

    def step(self, action) -> (GameState, bool, dict):
        if self.current_phase == Phase.DRAFT:
            self._act_on_draft(action)

            self._next_turn()

            if self.current_phase == Phase.DRAFT:
                self._new_draft_turn()
            elif self.current_phase == Phase.BATTLE:
                self._prepare_for_battle()

                self._new_battle_turn()

        elif self.current_phase == Phase.BATTLE:
            if action.type != ActionType.PASS:
                self._act_on_battle(action)
            else:
                self._next_turn()

                self._new_battle_turn()

        new_state = self._build_game_state()
        has_ended = False
        info = {'turn': self.turn}

        if self.players[PlayerOrder.FIRST].health <= 0:
            self.current_phase = Phase.ENDED
            info['winner'] = PlayerOrder.SECOND
            has_ended = True
        elif self.players[PlayerOrder.SECOND].health <= 0:
            self.current_phase = Phase.ENDED
            info['winner'] = PlayerOrder.FIRST
            has_ended = True

        info['phase'] = self.current_phase

        return new_state, has_ended, info

    def _next_turn(self) -> bool:
        if self.current_player == PlayerOrder.FIRST:
            self.current_player = PlayerOrder.SECOND

            return False
        else:
            self.current_player = PlayerOrder.FIRST
            self.turn += 1

            if self.turn > self.cards_in_deck \
                    and self.current_phase == Phase.DRAFT:
                self.current_phase = Phase.BATTLE
                self.turn = 1

            return True

    def _prepare_for_draft(self):
        """Prepare all game components for a draft phase"""
        self._draft_cards = self._new_draft()

        current_draft_choices = self._draft_cards[self.turn - 1]

        self.players = (Player(), Player())

        for player in self.players:
            player.hand = current_draft_choices

    def _prepare_for_battle(self):
        """Prepare all game components for a battle phase"""
        for player in self.players:
            player.hand = []
            player.lanes = ([], [])

            np.random.shuffle(player.deck)
            player.draw(4)
            player.base_mana = 0

        second_player = self.players[PlayerOrder.SECOND]
        second_player.draw()
        second_player.bonus_mana = 1

    def _new_draft_turn(self):
        """Initialize a draft turn"""
        current_draft_choices = self._draft_cards[self.turn - 1]

        for player in self.players:
            player.hand = current_draft_choices

    def _new_battle_turn(self):
        """Initialize a battle turn"""
        current_player = self.players[self.current_player]

        for creature in current_player.lanes[Lane.LEFT]:
            creature.can_attack = True
            creature.has_attacked_this_turn = False

        for creature in current_player.lanes[Lane.RIGHT]:
            creature.can_attack = True
            creature.has_attacked_this_turn = False

        if current_player.base_mana < 12:
            current_player.base_mana += 1

        current_player.mana = current_player.base_mana \
            + current_player.bonus_mana

        amount_to_draw = 1 + current_player.bonus_draw
        current_player.bonus_draw = 0

        if self.turn > 50:
            current_player.deck = []

        try:
            current_player.draw(amount_to_draw)
        except FullHandError:
            pass
        except EmptyDeckError:
            amount_of_damage = current_player.health \
                               - current_player.next_rune
            current_player.damage(amount_of_damage)

    def _act_on_draft(self, action):
        """Execute the action intended by the player in this draft turn"""
        current_player = self.players[self.current_player]

        chosen_index = action.origin if action.origin is not None else 0
        card = current_player.hand[chosen_index]

        current_player.deck.append(card.make_copy())

        current_player.actions.append(action)

    def _act_on_battle(self, action):
        """Execute the actions intended by the player in this battle turn"""
        current_player = self.players[self.current_player]

        try:
            if action.type == ActionType.SUMMON:
                self._do_summon(action)
            elif action.type == ActionType.ATTACK:
                self._do_attack(action)
            elif action.type == ActionType.USE:
                self._do_use(action)
            else:
                raise MalformedActionError("Invalid action type")

            current_player.actions.append(action)
        except (NotEnoughManaError, MalformedActionError, FullLaneError) as e:
            eprint("Action error:", e.message)

        for player in self.players:
            for lane in player.lanes:
                for creature in lane:
                    if creature.is_dead:
                        lane.remove(creature)

        if current_player.mana == 0:
            current_player.bonus_mana = 0

    def _do_summon(self, action):
        current_player = self.players[self.current_player]
        opposing_player = self.players[self.current_player.opposing()]

        if action.origin.cost > current_player.mana:
            raise NotEnoughManaError()

        if not isinstance(action.origin, Creature):
            raise MalformedActionError("Card being summoned is not a "
                                       "creature")

        if not isinstance(action.target, Lane):
            raise MalformedActionError("Target is not a lane")

        if len(current_player.lanes[action.target]) >= 3:
            raise FullLaneError()

        try:
            current_player.hand.remove(action.origin)
        except ValueError:
            raise MalformedActionError("Card is not in player's hand")

        action.origin.can_attack = False

        current_player.lanes[action.target].append(action.origin)

        current_player.bonus_draw += action.origin.card_draw
        current_player.health += action.origin.player_hp
        opposing_player.health += action.origin.enemy_hp

        current_player.mana -= action.origin.cost

    def _do_attack(self, action):
        current_player = self.players[self.current_player]
        opposing_player = self.players[self.current_player.opposing()]

        if not isinstance(action.origin, Creature):
            raise MalformedActionError("Attacking card is not a "
                                       "creature")

        if action.origin in current_player.lanes[Lane.LEFT]:
            origin_lane = Lane.LEFT
        elif action.origin in current_player.lanes[Lane.RIGHT]:
            origin_lane = Lane.RIGHT
        else:
            raise MalformedActionError("Attacking creature is not "
                                       "owned by player")

        guard_creatures = []

        for creature in opposing_player.lanes[origin_lane]:
            if creature.has_ability('G'):
                guard_creatures.append(creature)

        if len(guard_creatures) > 0:
            valid_targets = guard_creatures
        else:
            valid_targets = [None] + opposing_player.lanes[origin_lane]

        if action.target not in valid_targets:
            raise MalformedActionError("Invalid target")

        if not action.origin.able_to_attack():
            raise MalformedActionError("Attacking creature cannot "
                                       "attack")

        if action.target is None:
            damage_dealt = opposing_player.damage(action.origin.attack)

        elif isinstance(action.target, Creature):
            try:
                damage_dealt = action.target.damage(
                    action.origin.attack,
                    lethal=action.origin.has_ability('L'))

                action.origin.damage(
                    action.target.attack,
                    lethal=action.origin.has_ability('L')
                )

                excess_damage = action.origin.attack \
                                - action.target.defense

                if 'B' in action.origin.keywords and excess_damage > 0:
                    opposing_player.damage(excess_damage)

            except WardShieldError:
                damage_dealt = 0

        else:
            raise MalformedActionError("Target is not a creature or "
                                       "a player")

        if 'D' in action.origin.keywords:
            current_player.health += damage_dealt

        action.origin.has_attacked_this_turn = True

    def _do_use(self, action):
        current_player = self.players[self.current_player]
        opposing_player = self.players[self.current_player.opposing()]

        if action.origin.cost > current_player.mana:
            raise NotEnoughManaError()

        if action.target is not None and \
                not isinstance(action.target, Creature):
            error = "Target is not a creature or a player"
            raise MalformedActionError(error)

        try:
            current_player.hand.remove(action.origin)
        except ValueError:
            raise MalformedActionError("Card is not in player's hand")

        if isinstance(action.origin, GreenItem):
            is_own_creature = \
                action.target in current_player.lanes[Lane.LEFT] or \
                action.target in current_player.lanes[Lane.RIGHT]

            if action.target is None or not is_own_creature:
                error = "Green items should be used on friendly " \
                        "creatures"
                raise MalformedActionError(error)

            action.target.attack += action.origin.attack
            action.target.defense += action.origin.defense
            action.target.keywords = \
                action.target.keywords.union(action.origin.keywords)

            current_player.bonus_draw += action.target.card_draw
            current_player.health += action.target.player_hp
            opposing_player.health += action.target.enemy_hp

        elif isinstance(action.origin, RedItem):
            is_opp_creature = \
                action.target in opposing_player.lanes[Lane.LEFT] or \
                action.target in opposing_player.lanes[Lane.RIGHT]

            if action.target is None or not is_opp_creature:
                error = "Red items should be used on enemy " \
                        "creatures"
                raise MalformedActionError(error)

            action.target.attack += action.origin.attack
            action.target.defense += action.origin.defense
            action.target.keywords = \
                action.target.keywords.difference(
                    action.origin.keywords)

            current_player.bonus_draw += action.target.card_draw
            current_player.health += action.target.player_hp
            opposing_player.health += action.target.enemy_hp

        elif isinstance(action.origin, BlueItem):
            if isinstance(action.target, Creature):
                action.target.attack += action.origin.attack
                action.target.defense += action.origin.defense
                action.target.keywords = \
                    action.target.keywords.difference(
                        action.origin.keywords)
            elif action.target is None:
                opposing_player.damage(-action.origin.defense)
            else:
                raise MalformedActionError("Invalid target")

            current_player.bonus_draw += action.origin.card_draw
            current_player.health += action.origin.player_hp
            opposing_player.health += action.origin.enemy_hp

        else:
            error = "Card being used is not an item"
            raise MalformedActionError(error)

        current_player.mana -= action.origin.cost

    def _build_game_state(self) -> GameState:
        return GameState(self.current_phase, self.current_player,
                         self.turn, self.players)

    def _new_draft(self) -> List[List[Card]]:
        pool = np.random.choice(self._cards, 60, replace=False).tolist()
        draft = []

        for _ in range(self.cards_in_deck):
            draft.append(np.random.choice(pool, 3, replace=False).tolist())

        return draft

    @staticmethod
    def _load_cards() -> List[Card]:
        cards = []

        with open('gym_locm/cardlist.txt', 'r') as card_list:
            raw_cards = card_list.readlines()
            type_mapping = {'creature': Creature, 'itemRed': RedItem,
                            'itemGreen': GreenItem, 'itemBlue': BlueItem}

            for card in raw_cards:
                card_id, name, card_type, cost, attack, defense, \
                    keywords, player_hp, enemy_hp, card_draw, text = \
                    map(str.strip, card.split(';'))

                card_class = type_mapping[card_type]

                cards.append(card_class(int(card_id), name, card_type, int(cost),
                                        int(attack), int(defense), keywords,
                                        int(player_hp), int(enemy_hp),
                                        int(card_draw), text))

        assert len(cards) == 160

        return cards
