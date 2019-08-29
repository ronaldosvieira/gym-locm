from abc import ABC, abstractmethod

from gym_locm.engine import *

import pexpect


class Agent(ABC):
    @abstractmethod
    def act(self, state):
        pass


class PassBattleAgent(Agent):
    def act(self, state):
        return Action(ActionType.PASS)


class RandomBattleAgent(Agent):
    def act(self, state):
        return np.random.choice(state.available_actions)


class RuleBasedBattleAgent(Agent):
    def __init__(self):
        self.last_action = None

    def act(self, state):
        friends = state.current_player.lanes[0] + state.current_player.lanes[1]
        foes = state.opposing_player.lanes[0] + state.opposing_player.lanes[1]

        current_lane = list(Lane)[state.turn % 2]

        for card in state.current_player.hand:
            if isinstance(card, Creature) and card.cost <= state.current_player.mana\
                    and len(state.current_player.lanes[current_lane]) < 3:
                action = Action(ActionType.SUMMON, card, current_lane)

                return action

            elif isinstance(card, GreenItem) and card.cost <= state.current_player.mana\
                    and friends:
                return Action(ActionType.USE, card, friends[0])
            elif isinstance(card, RedItem) and card.cost <= state.current_player.mana\
                    and foes:
                return Action(ActionType.USE, card, foes[0])
            elif isinstance(card, BlueItem) and card.cost <= state.current_player.mana:
                return Action(ActionType.USE, card, None)

        for card in state.current_player.lanes[Lane.LEFT]:
            if card.can_attack and not card.has_attacked_this_turn:
                for enemy in state.opposing_player.lanes[Lane.LEFT]:
                    if enemy.has_ability('G'):
                        return Action(ActionType.ATTACK, card, enemy)

                return Action(ActionType.ATTACK, card, None)

        for card in state.current_player.lanes[Lane.RIGHT]:
            if card.can_attack and not card.has_attacked_this_turn:
                for enemy in state.opposing_player.lanes[Lane.RIGHT]:
                    if enemy.has_ability('G'):
                        return Action(ActionType.ATTACK, card, enemy)

                return Action(ActionType.ATTACK, card, None)

        return Action(ActionType.PASS)


class NativeAgent(Agent):
    def __init__(self, cmd):
        self._process = pexpect.spawn(cmd, echo=False, encoding='utf-8')

        self.action_buffer = []

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._process.terminate()

    @staticmethod
    def _encode_state(state):
        encoding = ""

        p, o = state.current_player, state.opposing_player

        for cp in p, o:
            to_draw = 0 if state.phase == Phase.DRAFT else 1 + cp.bonus_draw

            encoding += f"{cp.health} {cp.mana} {len(cp.deck)} " \
                        f"{cp.next_rune} {to_draw}\n"

        op_hand = 0 if state.phase == Phase.DRAFT else len(o.hand)
        last_actions = []

        for action in reversed(o.actions):
            if action.type == ActionType.PASS:
                break

            last_actions.append(action)

        encoding += f"{op_hand} {len(last_actions)}\n"

        for a in last_actions:
            names = {
                ActionType.USE: 'USE',
                ActionType.SUMMON: 'SUMMON',
                ActionType.ATTACK: 'ATTACK',
            }

            target_id = -1 if a.target is None else a.target.instance_id

            encoding += f"{a.origin.id} {names[a.type]} " \
                        f"{a.origin.instance_id} {target_id}\n"

        cards = p.hand + p.lanes[0] + p.lanes[1] + o.lanes[0] + o.lanes[1]

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

            if c.type == 'creature':
                c.cardType = 0
            elif c.type == 'itemGreen':
                c.cardType = 1
            elif c.type == 'itemRed':
                c.cardType = 2
            elif c.type == 'itemBlue':
                c.cardType = 3

            abilities = list('------')

            for i, a in enumerate(list('BCDGLW')):
                if c.has_ability(a):
                    abilities[i] = a

            c.abilities = "".join(abilities)

            c.instance_id = -1 if c.instance_id is None else c.instance_id

        for i, c in enumerate(cards):
            encoding += f"{c.id} {c.instance_id} {c.location} {c.cardType} " \
                        f"{c.cost} {c.attack} {c.defense} {c.abilities} " \
                        f"{c.player_hp} {c.enemy_hp} {c.card_draw} {c.lane}\n"

        return encoding

    @staticmethod
    def _decode_actions(state, actions):
        actions = actions.split(';')
        decoded_actions = []

        cp_lanes = state.current_player.lanes[0] + state.current_player.lanes[1]
        op_lanes = state.opposing_player.lanes[0] + state.opposing_player.lanes[1]

        hands = state.current_player.hand + state.opposing_player.hand

        cards = hands + cp_lanes + op_lanes

        def _find_card(instance_id):
            if instance_id == -1:
                return None

            for card in cards:
                if card.instance_id == instance_id:
                    return card

        for action in actions:
            tokens = action.split()

            if not tokens:
                continue

            if tokens[0] == 'PASS':
                decoded_actions.append(Action(ActionType.PASS))
            elif tokens[0] == 'PICK':
                decoded_actions.append(Action(ActionType.PICK, int(tokens[1])))
            elif tokens[0] == 'USE':
                origin = _find_card(int(tokens[1]))
                target = _find_card(int(tokens[2]))

                decoded_actions.append(Action(ActionType.USE, origin, target))
            elif tokens[0] == 'SUMMON':
                origin = _find_card(int(tokens[1]))
                target = Lane(int(tokens[2]))

                decoded_actions.append(Action(ActionType.SUMMON, origin, target))
            elif tokens[0] == 'ATTACK':
                origin = _find_card(int(tokens[1]))
                target = _find_card(int(tokens[2]))

                decoded_actions.append(Action(ActionType.ATTACK, origin, target))

        return decoded_actions

    def act(self, state):
        if self.action_buffer:
            return self.action_buffer.pop()

        self._process.write(self._encode_state(state))

        actions = []

        while not actions:
            actions = self._process.readline()

            actions = list(reversed(self._decode_actions(state, actions)))

        if actions[-1].type != ActionType.PASS and state.phase != Phase.DRAFT:
            actions = [Action(ActionType.PASS)] + actions

        self.action_buffer = actions

        return self.action_buffer.pop()


PassDraftAgent = PassBattleAgent
RandomDraftAgent = RandomBattleAgent


class RuleBasedDraftAgent(Agent):
    def act(self, state):
        for i, card in enumerate(state.current_player.hand):
            if isinstance(card, Creature) and card.has_ability('G'):
                return Action(ActionType.PICK, i)

        return Action(ActionType.PICK, 0)


class IceboxDraftAgent(Agent):
    @staticmethod
    def _icebox_eval(card):
        value = card.attack + card.defense

        value -= 6.392651 * 0.001 * (card.cost ** 2)
        value -= 1.463006 * card.cost
        value -= 1.435985

        value += 5.985350469 * 0.01 * ((card.player_hp - card.enemy_hp) ** 2)
        value += 3.880957 * 0.1 * (card.player_hp - card.enemy_hp)
        value += 5.219

        value -= 5.516179907 * (card.card_draw ** 2)
        value += 0.239521 * card.card_draw
        value -= 1.63766 * 0.1

        value -= 7.751401869 * 0.01

        if 'B' in card.keywords:
            value += 0.0
        if 'C' in card.keywords:
            value += 0.26015517
        if 'D' in card.keywords:
            value += 0.15241379
        if 'G' in card.keywords:
            value += 0.04418965
        if 'L' in card.keywords:
            value += 0.15313793
        if 'W' in card.keywords:
            value += 0.16238793

        return value

    def act(self, state):
        return np.argmax(list(map(self._icebox_eval, state.current_player.hand)))


class ClosetAIDraftAgent(Agent):
    scores = [
        -666,   65,   50,   80,   50,   70,   71,  115,   71,   73,
        43,   77,   62,   63,   50,   66,   60,   66,   90,   75,
        50,   68,   67,  100,   42,   63,   67,   52,   69,   90,
        60,   47,   87,   81,   67,   62,   75,   94,   56,   62,
        51,   61,   43,   54,   97,   64,   67,   49,  109,  111,
        89,  114,   93,   92,   89,    2,   54,   25,   63,   76,
        58,   99,   79,   19,   82,  115,  106,  104,  146,   98,
        70,   56,   65,   52,   54,   65,   55,   77,   48,   84,
        115,   75,   89,   68,   80,   71,   46,   73,   69,   47,
        63,   70,   11,   71,   54,   85,   77,   77,   64,   82,
        62,   49,   43,   78,   67,   72,   67,   36,   48,   75,
        -8,   82,   69,   32,   87,   98,  124,   35,   60,   59,
        49,   72,   54,   35,   22,   50,   54,   51,   54,   59,
        38,   31,   43,   62,   55,   57,   41,   70,   38,   76,
        1, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100
    ]

    def _closet_ai_eval(self, card):
        return self.scores[card.id - 1]

    def act(self, state):
        return np.argmax(list(map(self._closet_ai_eval, state.current_player.hand)))
