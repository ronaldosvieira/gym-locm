import copy
import os
import sys
from operator import attrgetter

from gym_locm.engine.enums import *
from gym_locm.engine.phases import *
from gym_locm.exceptions import *


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class Player:
    def __init__(self, player_id):
        self.id = player_id

        self.health = 30
        self.base_mana = 0
        self.bonus_mana = 0
        self.mana = 0
        self.next_rune = 25
        self.bonus_draw = 0

        self.last_drawn = 0

        self.deck = []
        self.hand = []
        self.lanes = ([], [])

        self.actions = []

    def draw(self, amount: int = 1):
        for i in range(amount):
            if len(self.deck) == 0:
                raise EmptyDeckError(amount - i)

            if len(self.hand) >= 8:
                raise FullHandError()

            self.hand.append(self.deck.pop())


class Card:
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
        self.keywords = set(list(keywords.replace("-", "")))
        self.player_hp = player_hp
        self.enemy_hp = enemy_hp
        self.card_draw = card_draw
        self.area = area
        self.text = text

    def has_ability(self, keyword: str) -> bool:
        return keyword in self.keywords

    def make_copy(self, instance_id=None) -> "Card":
        cloned_card = copy.deepcopy(self)

        if instance_id is not None:
            cloned_card.instance_id = instance_id
        else:
            cloned_card.instance_id = None

        return cloned_card

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

    @staticmethod
    def mockup_card():
        return Card(0, "", 0, 0, 0, 0, "------", 0, 0, 0, 0, "", instance_id=None)


class Creature(Card):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_dead = False
        self.can_attack = False
        self.has_attacked_this_turn = False

    def remove_ability(self, ability: str):
        self.keywords.discard(ability)

    def add_ability(self, ability: str):
        self.keywords.add(ability)

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
    pass


class GreenItem(Item):
    pass


class RedItem(Item):
    pass


class BlueItem(Item):
    pass


class Action:
    def __init__(self, action_type, origin=None, target=None):
        self.type = action_type
        self.origin = origin
        self.target = target

    def __eq__(self, other):
        return (
            other is not None
            and self.type == other.type
            and self.origin == other.origin
            and self.target == other.target
        )

    def __repr__(self):
        return f"{self.type} {self.origin} {self.target}"


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

    return cards


_cards = load_cards()


def get_locm12_card_list():
    return _cards


class State:
    def __init__(
        self,
        seed=None,
        version="1.5",
        items=True,
        deck_building_kwargs=None,
        battle_kwargs=None,
    ):

        self.rng = np.random.default_rng(seed=seed)
        self.items = items
        self.version = version
        self.turn = 1
        self.was_last_action_invalid = False

        if version == "1.5":
            self.deck_building_phase = ConstructedPhase(self, self.rng, items)
            self.battle_phase = Version15BattlePhase(self, self.rng, items)

            self.phase = Phase.CONSTRUCTED

        elif version == "1.2":
            self.deck_building_phase = DraftPhase(
                self, self.rng, items, **deck_building_kwargs
            )
            self.battle_phase = Version12BattlePhase(self, self.rng, items)

            self.phase = Phase.DRAFT

        else:
            raise ValueError(
                f'Invalid version {version}. Supported versions: "1.5" and "1.2"'
            )

        self._phase = self.deck_building_phase
        self._phase.prepare()

        self.players = (Player(PlayerOrder.FIRST), Player(PlayerOrder.SECOND))

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
        return copy.deepcopy(self)

    def __str__(self) -> str:
        encoding = ""

        p, o = self.current_player, self.opposing_player

        for cp in p, o:
            draw = cp.last_drawn if cp == self.current_player else 1 + cp.bonus_draw

            if self.version == "1.5":
                encoding += f"{cp.health} {cp.base_mana + cp.bonus_mana} {len(cp.deck)} {draw}\n"
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
    def from_native_input(game_input):
        game_input = iter(game_input)

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
        cp.deck = [Card.mockup_card() for _ in range(deck)]

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
        op.deck = [Card.mockup_card() for _ in range(deck)]

        if mana != 0:
            state.phase = Phase.BATTLE
            state._phase = state.battle_phase
        else:
            state.turn = deck + 1

        opp_hand, opp_actions = map(int, next(game_input).split())

        state.opposing_player.hand = [Card.mockup_card() for _ in range(opp_hand)]

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

        return state


Game = State
