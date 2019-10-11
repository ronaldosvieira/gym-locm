from abc import ABC, abstractmethod

from gym_locm.engine import *
from gym_locm.algorithms import MCTS

import pexpect
import time
import random


class Agent(ABC):
    @abstractmethod
    def seed(self, seed):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, state):
        pass


class PassBattleAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

    def act(self, state):
        return Action(ActionType.PASS)


class RandomBattleAgent(Agent):
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def seed(self, seed):
        self.random.seed(seed)

    def reset(self):
        pass

    def act(self, state):
        index = int(len(state.available_actions) * random.random())

        return state.available_actions[index]


class RuleBasedBattleAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

    def act(self, state):
        friends = state.current_player.lanes[0] + state.current_player.lanes[1]
        foes = state.opposing_player.lanes[0] + state.opposing_player.lanes[1]

        current_lane = list(Lane)[state.turn % 2]

        for card in state.current_player.hand:
            origin = card.instance_id

            if isinstance(card, Creature) and card.cost <= state.current_player.mana\
                    and len(state.current_player.lanes[current_lane]) < 3:
                action = Action(ActionType.SUMMON, origin, current_lane)

                return action

            elif isinstance(card, GreenItem) and card.cost <= state.current_player.mana\
                    and friends:
                target = friends[0].instance_id

                return Action(ActionType.USE, origin, target)
            elif isinstance(card, RedItem) and card.cost <= state.current_player.mana\
                    and foes:
                target = foes[0].instance_id

                return Action(ActionType.USE, origin, target)
            elif isinstance(card, BlueItem) and card.cost <= state.current_player.mana:
                return Action(ActionType.USE, origin, None)

        for card in state.current_player.lanes[Lane.LEFT]:
            origin = card.instance_id

            if card.can_attack and not card.has_attacked_this_turn:
                for enemy in state.opposing_player.lanes[Lane.LEFT]:
                    if enemy.has_ability('G'):
                        target = enemy.instance_id

                        return Action(ActionType.ATTACK, origin, target)

                return Action(ActionType.ATTACK, card, None)

        for card in state.current_player.lanes[Lane.RIGHT]:
            origin = card.instance_id

            if card.can_attack and not card.has_attacked_this_turn:
                for enemy in state.opposing_player.lanes[Lane.RIGHT]:
                    if enemy.has_ability('G'):
                        target = enemy.instance_id

                        return Action(ActionType.ATTACK, origin, target)

                return Action(ActionType.ATTACK, origin, None)

        return Action(ActionType.PASS)


class NativeAgent(Agent):
    action_buffer = []

    def __init__(self, cmd):
        self._process = pexpect.spawn(cmd, echo=False, encoding='utf-8')

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._process.terminate()

    def seed(self, seed):
        pass

    def reset(self):
        self.action_buffer = []

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
    def _decode_actions(actions):
        actions = actions.split(';')
        decoded_actions = []

        for action in actions:
            tokens = action.split()

            if not tokens:
                continue

            if tokens[0] == 'PASS':
                decoded_actions.append(Action(ActionType.PASS))
            elif tokens[0] == 'PICK':
                decoded_actions.append(Action(ActionType.PICK, int(tokens[1])))
            elif tokens[0] == 'USE':
                origin = int(tokens[1])
                target = int(tokens[2])
                target = target if target >= 0 else None

                decoded_actions.append(Action(ActionType.USE, origin, target))
            elif tokens[0] == 'SUMMON':
                origin = int(tokens[1])
                target = Lane(int(tokens[2]))

                decoded_actions.append(Action(ActionType.SUMMON, origin, target))
            elif tokens[0] == 'ATTACK':
                origin = int(tokens[1])
                target = int(tokens[2])
                target = target if target >= 0 else None

                decoded_actions.append(Action(ActionType.ATTACK, origin, target))

        return decoded_actions

    def act(self, state):
        if self.action_buffer:
            return self.action_buffer.pop()

        self._process.write(self._encode_state(state))

        actions = []

        while not actions:
            actions = self._process.readline()

            actions = list(reversed(self._decode_actions(actions)))

        if actions[0].type != ActionType.PASS and state.phase != Phase.DRAFT:
            actions = [Action(ActionType.PASS)] + actions

        self.action_buffer = actions

        return self.action_buffer.pop()


class MCTSBattleAgent(Agent):
    def __init__(self, agents=(RandomBattleAgent(), RandomBattleAgent())):
        self.agents = agents

    def seed(self, seed):
        for agent in self.agents:
            agent.seed(seed)

    def reset(self):
        pass

    def act(self, state, time_limit_ms=200):
        searcher = MCTS(agents=self.agents)

        if len(state.available_actions) == 1:
            return state.available_actions[0]

        start_time = int(time.time() * 1000.0)

        while True:
            current_time = int(time.time() * 1000.0)

            if current_time - start_time > time_limit_ms:
                break

            searcher.do_rollout(state)


        action = searcher.choose(state)

        return action


class CoacBattleAgent(Agent):
    def __init__(self):
        self.time_limit_ms = float("-inf")
        self.start_time = None

        self.player = None
        self.leaf = 0

    def seed(self, seed):
        pass

    def reset(self):
        self.leaf = 0

    @staticmethod
    def _eval_creature(creature):
        score = 0

        if creature.attack > 0:
            score += 20
            score += creature.attack * 10
            score += creature.defense * 5

            if creature.has_ability('W'):
                score += creature.attack * 5

            if creature.has_ability('L'):
                score += 20

        if creature.has_ability('G'):
            score += 9

        return score

    @staticmethod
    def _eval_state(state):
        score = 0

        player, enemy = state.current_player, state.opposing_player

        for lane in player.lanes:
            for creature in lane:
                score += CoacBattleAgent._eval_creature(creature)

        for lane in enemy.lanes:
            for creature in lane:
                score -= CoacBattleAgent._eval_creature(creature)

        for card in player.hand:
            if not isinstance(card, Creature):
                score += 21  # todo: discover what passed means

        if len(player.hand) + player.bonus_draw + 1 <= 8:
            score += (player.bonus_draw + 1) * 5

        score += player.health * 2
        score -= enemy.health * 2

        if player.health < 5:
            score -= 100

        if enemy.health <= 0:
            score += 100000
        elif player.health <= 0:
            score -= 100000

        return score

    @staticmethod
    def _check_lethal(state):
        damage = 0
        ref_dict = {}

        for lane in state.current_player.lanes:
            for creature in lane:
                ref_dict[creature.instance_id] = creature

        for action in state.available_actions:
            if action.type == ActionType.ATTACK and action.target is None:
                damage += ref_dict[action.origin].attack

        return damage >= state.opposing_player.health

    @staticmethod
    def all_attack_enemy_player(state):
        for action in state.available_actions:
            if action.type == ActionType.ATTACK and action.target is None:
                return action

    def _brute_force_leaf(self, state, alpha):
        enemy_depth = 1
        best_action = Action(ActionType.PASS)

        state.act(best_action)

        if state.current_player.id != self.player:
            _, score = self._run_brute_force(state, enemy_depth, alpha)

            return best_action, score

        self.leaf += 1

        return best_action, -self._eval_state(state)

    def _brute_force(self, state, depth, alpha):
        state = state.clone()

        if depth <= 0:
            return self._brute_force_leaf(state, alpha)

        legal_moves = state.available_actions

        if legal_moves[0].type == ActionType.PASS:
            return self._brute_force_leaf(state, alpha)

        if len(state.available_actions) >= 35:
            depth -= 1

        current_time = int(time.time() * 1000.0)

        if current_time - self.start_time >= self.time_limit_ms:
            depth = 1

        best_score = float("-inf")
        best_action = Action(ActionType.PASS)

        for action in state.available_actions:
            if action.type == ActionType.PASS:
                continue

            state_copy = state.clone()

            state_copy.act(action)

            if state_copy.winner is not None:
                score = 100000 if state_copy.winner == self.player else -100000

                return action, score

            _, score = self._brute_force(state_copy, depth - 1, alpha)

            if state.current_player.id != self.player and alpha > -score:
                return action, score

            if state.current_player.id == self.player:
                alpha = alpha if alpha > score else score

            if score > best_score:
                best_action = action
                best_score = score

        return best_action, best_score

    def _run_brute_force(self, state, depth, alpha):
        action, score = self._brute_force(state, depth, alpha)

        if self._check_lethal(state):
            action = self.all_attack_enemy_player(state)

            if state.current_player.id == self.player:
                return action, 10000
            else:
                return action, -100000

        return action, self._eval_state(state)

    def act(self, state, time_limit_ms=1000):
        self.leaf = 0

        self.start_time = int(time.time() * 1000.0)
        self.player = state.current_player.id

        if self._check_lethal(state):
            action = self.all_attack_enemy_player(state)
        else:
            depth = 3

            action, _ = self._run_brute_force(state, depth, float("-inf"))

        return action


PassDraftAgent = PassBattleAgent
RandomDraftAgent = RandomBattleAgent


class RuleBasedDraftAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

    def act(self, state):
        for i, card in enumerate(state.current_player.hand):
            if isinstance(card, Creature) and card.has_ability('G'):
                return Action(ActionType.PICK, i)

        return Action(ActionType.PICK, 0)


class IceboxDraftAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

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
        index = np.argmax(list(map(self._icebox_eval, state.current_player.hand)))

        return Action(ActionType.PICK, index)


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

    def seed(self, seed):
        pass

    def reset(self):
        pass

    def _closet_ai_eval(self, card):
        return self.scores[card.id - 1]

    def act(self, state):
        index = np.argmax(list(map(self._closet_ai_eval, state.current_player.hand)))

        return Action(ActionType.PICK, index)


class CoacDraftAgent(Agent):
    scores = {
        'p1': [
            68, 7, 65, 49, 116, 69, 151, 48, 53, 51,
            44, 67, 29, 139, 84, 18, 158, 28, 64, 80,
            33, 85, 32, 147, 103, 37, 54, 52, 50, 99,
            23, 87, 66, 81, 148, 88, 150, 121, 82, 95,
            115, 133, 152, 19, 109, 157, 105, 3, 75, 96,
            114, 9, 106, 144, 129, 17, 111, 128, 12, 11,
            145, 15, 21, 8, 134, 155, 141, 70, 90, 135,
            104, 41, 112, 61, 5, 97, 26, 34, 73, 6,
            36, 86, 77, 83, 13, 89, 79, 93, 149, 59,
            159, 74, 94, 38, 98, 126, 39, 30, 137, 100,
            62, 122, 22, 72, 118, 1, 47, 71, 4, 91,
            27, 56, 119, 101, 45, 16, 146, 58, 120, 142,
            127, 25, 108, 132, 40, 14, 76, 125, 102, 131,
            123, 2, 35, 130, 107, 43, 63, 31, 138, 124,
            154, 78, 46, 24, 10, 136, 113, 60, 57, 92,
            117, 42, 55, 153, 20, 156, 143, 110, 160, 140
        ],
        'p1_creature': [
            68, 7, 65, 49, 116, 69, 151, 48, 53, 51,
            44, 67, 29, 139, 84, 18, 158, 28, 64, 80,
            33, 85, 32, 147, 103, 37, 54, 52, 50, 99,
            23, 87, 66, 81, 148, 88, 150, 121, 82, 95,
            115, 133, 152, 19, 109, 157, 105, 3, 75, 96,
            114, 9, 106, 144, 129, 17, 111, 128, 12, 11,
            145, 15, 21, 8, 134, 155, 141, 70, 90, 135,
            104, 41, 112, 61, 5, 97, 26, 34, 73, 6,
            36, 86, 77, 83, 13, 89, 79, 93, 149, 59,
            159, 74, 94, 38, 98, 126, 39, 30, 137, 100,
            62, 122, 22, 72, 118, 1, 47, 71, 4, 91,
            27, 56, 119, 101, 45, 16, 146, 58, 120, 142,
            127, 25, 108, 132, 40, 14, 76, 125, 102, 131,
            123, 2, 35, 130, 107, 43, 63, 31, 138, 124,
            154, 78, 46, 24, 10, 136, 113, 60, 57, 92,
            117, 42, 55, 153, 20, 156, 143, 110, 160, 140
        ],
        'p2': [
            68, 7, 65, 49, 116, 69, 151, 48, 53, 51,
            44, 67, 29, 139, 84, 18, 158, 28, 64, 80,
            33, 85, 32, 147, 103, 37, 54, 52, 50, 99,
            23, 87, 66, 81, 148, 88, 150, 121, 82, 95,
            115, 133, 152, 19, 109, 157, 105, 3, 75, 96,
            114, 9, 106, 144, 129, 17, 111, 128, 12, 15,
            11, 145, 21, 8, 134, 155, 141, 70, 90, 135,
            104, 112, 41, 61, 5, 97, 26, 34, 73, 6,
            36, 86, 77, 83, 89, 13, 79, 93, 59, 149,
            159, 74, 94, 38, 126, 98, 39, 30, 100, 62,
            137, 122, 22, 72, 118, 1, 47, 71, 4, 91,
            56, 27, 119, 101, 45, 146, 16, 120, 58, 142,
            25, 127, 108, 132, 40, 14, 76, 125, 102, 123,
            131, 2, 35, 130, 107, 43, 63, 31, 138, 124,
            154, 78, 46, 24, 10, 136, 113, 60, 57, 92,
            117, 55, 42, 153, 20, 156, 143, 110, 160, 140
        ],
        'p2_creature': [
            68, 7, 65, 49, 116, 69, 151, 48, 53, 51,
            44, 67, 29, 139, 84, 18, 158, 28, 64, 80,
            33, 85, 32, 147, 103, 37, 54, 52, 50, 99,
            23, 87, 66, 81, 148, 88, 150, 121, 82, 95,
            115, 133, 152, 19, 109, 157, 105, 3, 75, 96,
            114, 9, 106, 144, 129, 17, 111, 128, 12, 15,
            11, 145, 21, 8, 134, 155, 141, 70, 90, 135,
            104, 112, 41, 61, 5, 97, 26, 34, 73, 6,
            36, 86, 77, 83, 89, 13, 79, 93, 59, 149,
            159, 74, 94, 38, 126, 98, 39, 30, 100, 62,
            137, 122, 22, 72, 118, 1, 47, 71, 4, 91,
            56, 27, 119, 101, 45, 146, 16, 120, 58, 142,
            25, 127, 108, 132, 40, 14, 76, 125, 102, 123,
            131, 2, 35, 130, 107, 43, 63, 31, 138, 124,
            154, 78, 46, 24, 10, 136, 113, 60, 57, 92,
            117, 55, 42, 153, 20, 156, 143, 110, 160, 140
        ]
    }

    def seed(self, seed):
        pass

    def __init__(self):
        self.creatures_drafted = 0
        self.drafted = 0

    def reset(self):
        self.drafted = 0
        self.creatures_drafted = 0

    def act(self, state):
        key = 'p1' if state.current_player.id == PlayerOrder.FIRST else 'p2'

        if self.drafted > 0 and self.creatures_drafted / self.drafted < 0.4:
            key += '_creature'

        def _coac_eval(card):
            return self.scores[key].index(card.id)

        chosen_card = np.argmin(list(map(_coac_eval, state.current_player.hand)))

        self.drafted += 1

        if isinstance(state.current_player.hand[chosen_card], Creature):
            self.creatures_drafted += 1

        return Action(ActionType.PICK, chosen_card)


class RLDraftAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

    def __init__(self, algorithm, model='draft-trpo-3kk'):
        self.model = algorithm.load("models/" + model)

    def act(self, state):
        action, _ = self.model.predict(state)

        return Action(ActionType.PICK, action[0])
