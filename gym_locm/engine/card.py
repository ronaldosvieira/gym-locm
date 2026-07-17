import os
from typing import List

from gym_locm.exceptions import WardShieldError

# Keyword bitfield constants — enables O(1) keyword operations via bitwise ops
BREAKTHROUGH = 1   # B
CHARGE       = 2   # C
DRAIN        = 4   # D
GUARD        = 8   # G
LETHAL       = 16  # L
WARD         = 32  # W

KEYWORD_BITS = {'B': BREAKTHROUGH, 'C': CHARGE, 'D': DRAIN, 'G': GUARD, 'L': LETHAL, 'W': WARD}
KEYWORD_CHARS = tuple('BCDGLW')


def _parse_keywords(keyword_str: str) -> int:
    """Convert a keyword string like 'BCG' or 'B-D--W' into a bitfield integer."""
    bits = 0
    for ch in keyword_str:
        if ch in KEYWORD_BITS:
            bits |= KEYWORD_BITS[ch]
    return bits


class Card:
    __slots__ = (
        'id', 'instance_id', 'name', 'type', 'cost', 'attack', 'defense',
        'keywords', 'player_hp', 'enemy_hp', 'card_draw', 'area', 'text',
        # Dynamic attrs assigned by game_state.py __str__ and phases.py:
        'location', 'lane', 'cardType', 'abilities',
    )
    def __init__(
        self,
        card_id,
        name,
        card_type,
        cost,
        attack,
        defense,
        keywords,
        player_hp,
        enemy_hp,
        card_draw,
        area,
        text,
        instance_id=None,
    ):
        self.id = card_id
        self.instance_id = instance_id
        self.name = name
        self.type = card_type
        self.cost = cost
        self.attack = attack
        self.defense = defense
        self.keywords = _parse_keywords(keywords) if isinstance(keywords, str) else keywords
        self.player_hp = player_hp
        self.enemy_hp = enemy_hp
        self.card_draw = card_draw
        self.area = area
        self.text = text

    def has_ability(self, keyword: str) -> bool:
        return bool(self.keywords & KEYWORD_BITS[keyword])

    def __eq__(self, other):
        return (
            other is not None
            and self.instance_id is not None
            and other.instance_id is not None
            and self.instance_id == other.instance_id
        )

    def __repr__(self):
        if self.name:
            return f"({self.instance_id}: {self.name})"
        else:
            return f"({self.instance_id})"

    def make_copy(self, instance_id=None) -> "Card":
        cloned_card = Card.empty_copy(self)

        cloned_card.id = self.id
        cloned_card.name = self.name
        cloned_card.type = self.type
        cloned_card.cost = self.cost
        cloned_card.attack = self.attack
        cloned_card.defense = self.defense
        cloned_card.keywords = self.keywords  # int is immutable, no copy needed
        cloned_card.player_hp = self.player_hp
        cloned_card.enemy_hp = self.enemy_hp
        cloned_card.card_draw = self.card_draw
        cloned_card.area = self.area
        cloned_card.text = self.text

        if instance_id is not None:
            cloned_card.instance_id = instance_id
        else:
            cloned_card.instance_id = None

        return cloned_card

    @staticmethod
    def empty_copy(card):
        class Empty(type(card)):
            __slots__ = ()
            def __init__(self):
                pass

        new_copy = Empty()
        new_copy.__class__ = type(card)

        return new_copy

    @staticmethod
    def mockup_card():
        return Card(0, "", 0, 0, 0, 0, "------", 0, 0, 0, 0, "", instance_id=None)


class Creature(Card):
    __slots__ = ('is_dead', 'can_attack', 'has_attacked_this_turn', 'summon_counter')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_dead = False
        self.can_attack = False
        self.has_attacked_this_turn = False

    def remove_ability(self, ability: str):
        self.keywords &= ~KEYWORD_BITS[ability]

    def add_ability(self, ability: str):
        self.keywords |= KEYWORD_BITS[ability]

    def able_to_attack(self) -> bool:
        return not self.has_attacked_this_turn and (
            self.can_attack or self.has_ability("C")
        )

    def damage(self, amount: int = 1, lethal: bool = False) -> int:
        if amount <= 0:
            return 0

        if self.has_ability("W"):
            self.remove_ability("W")

            raise WardShieldError()

        self.defense -= amount

        if lethal or self.defense <= 0:
            self.is_dead = True

        return amount

    def make_copy(self, instance_id=None) -> "Card":
        cloned_card = super().make_copy(instance_id)

        cloned_card.is_dead = self.is_dead
        cloned_card.can_attack = self.can_attack
        cloned_card.has_attacked_this_turn = self.has_attacked_this_turn

        return cloned_card


class Item(Card):
    __slots__ = ()


class GreenItem(Item):
    __slots__ = ()


class RedItem(Item):
    __slots__ = ()


class BlueItem(Item):
    __slots__ = ()


def load_cards() -> List["Card"]:
    """
    Read the LOCM 1.2 card list.
    Available at
    https://github.com/acatai/Strategy-Card-Game-AI-Competition/blob/master/referee1.5-java/src/main/resources/cardlist.txt
    """
    cards = []

    with open(os.path.dirname(__file__) + "/resources/cardlist.txt", "r") as card_list:
        raw_cards = card_list.readlines()
        type_mapping = {
            "creature": (Creature, 0),
            "itemGreen": (GreenItem, 1),
            "itemRed": (RedItem, 2),
            "itemBlue": (BlueItem, 3),
        }

        for card in raw_cards:
            (
                card_id,
                name,
                card_type,
                cost,
                attack,
                defense,
                keywords,
                player_hp,
                enemy_hp,
                card_draw,
                text,
            ) = map(str.strip, card.split(";"))

            card_class, type_id = type_mapping[card_type]

            cards.append(
                card_class(
                    int(card_id),
                    name,
                    type_id,
                    int(cost),
                    int(attack),
                    int(defense),
                    keywords,
                    int(player_hp),
                    int(enemy_hp),
                    int(card_draw),
                    0,
                    text,
                )
            )

    assert len(cards) == 160

    return tuple(cards)


_cards = None


def get_locm12_card_list():
    global _cards

    if _cards is None:
        _cards = load_cards()

    return _cards
