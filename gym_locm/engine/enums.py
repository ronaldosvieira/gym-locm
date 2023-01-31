from enum import IntEnum, Enum


class Phase(IntEnum):
    DECK_BUILDING = 0
    DRAFT = 0
    CONSTRUCTED = 0
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

    def opposing(self):
        return Lane((self + 1) % 2)


class ActionType(Enum):
    PICK = 0
    CHOOSE = 1
    SUMMON = 2
    ATTACK = 3
    USE = 4
    PASS = 5


class Area(IntEnum):
    NONE = 0
    TYPE_1 = 1
    TYPE_2 = 2


class Location(IntEnum):
    PLAYER_HAND = 0
    ENEMY_HAND = 1

    PLAYER_BOARD = 10
    PLAYER_LEFT_LANE = 10
    PLAYER_RIGHT_LANE = 11

    ENEMY_BOARD = 20
    ENEMY_LEFT_LANE = 20
    ENEMY_RIGHT_LANE = 21


class DamageSource(Enum):
    GAME = 0
    SELF = 1
    OPPONENT = 2
