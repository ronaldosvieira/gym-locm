from abc import ABC, abstractmethod
from operator import attrgetter
from typing import Type

from pexpect import TIMEOUT, EOF

from gym_locm.engine import *
from gym_locm.algorithms import MCTS

import pexpect
import time
import random

from gym_locm.util import is_it


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


class GreedyBattleAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

    @staticmethod
    def eval_state(state):
        score = 0

        pl = state.current_player
        op = state.opposing_player

        # check opponent's death
        if op.health <= 0:
            score += 1000

        # check own death
        if pl.health <= 0:
            score -= 1000

        # health difference
        score += (pl.health - op.health) * 2

        for pl_lane, op_lane in zip(pl.lanes, op.lanes):
            # card count
            score += (len(pl_lane) - len(pl_lane)) * 10

            # card strength
            score += sum(c.attack + c.defense for c in pl_lane)
            score -= sum(c.attack + c.defense for c in op_lane)

        return score

    def act(self, state):
        best_action, best_score = Action(ActionType.PASS), float("-inf")

        for action in state.available_actions:

            if action.type == ActionType.PASS:
                continue

            state_copy = state.clone()
            state_copy.act(action)

            score = self.eval_state(state_copy)

            if score > best_score:
                best_action, best_score = action, score

        return best_action


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

            if isinstance(card, Creature) and card.cost <= state.current_player.mana \
                    and len(state.current_player.lanes[current_lane]) < 3:
                action = Action(ActionType.SUMMON, origin, current_lane)

                return action

            elif isinstance(card, GreenItem) and card.cost <= state.current_player.mana \
                    and friends:
                target = friends[0].instance_id

                return Action(ActionType.USE, origin, target)
            elif isinstance(card, RedItem) and card.cost <= state.current_player.mana \
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

                return Action(ActionType.ATTACK, origin, None)

        for card in state.current_player.lanes[Lane.RIGHT]:
            origin = card.instance_id

            if card.can_attack and not card.has_attacked_this_turn:
                for enemy in state.opposing_player.lanes[Lane.RIGHT]:
                    if enemy.has_ability('G'):
                        target = enemy.instance_id

                        return Action(ActionType.ATTACK, origin, target)

                return Action(ActionType.ATTACK, origin, None)

        return Action(ActionType.PASS)


class MaxAttackBattleAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

    def act(self, state):
        lanes = zip(list(Lane),
                    state.current_player.lanes,
                    state.opposing_player.lanes)

        for lane, friends, foes in lanes:
            guard_foes = filter(lambda c: c.has_ability('G'), foes)

            friends = filter(Creature.able_to_attack, friends)
            friends = sorted(friends, key=attrgetter('attack'), reverse=True)

            for creature in friends:
                try:
                    target = next(guard_foes)
                except StopIteration:
                    target = None

                return Action(ActionType.ATTACK, creature.instance_id, target)

        creatures_in_hand = filter(is_it(Creature),
                                   state.current_player.hand)
        creatures_in_hand = filter(has_enough_mana(state.current_player.mana),
                                   creatures_in_hand)
        creatures_in_hand = sorted(creatures_in_hand,
                                   key=attrgetter('attack'),
                                   reverse=True)

        lanes = (l for l in Lane if len(state.current_player.lanes[l]) < 3)

        try:
            for creature in creatures_in_hand:
                return Action(ActionType.SUMMON, creature.instance_id, next(lanes))
        except StopIteration:
            pass

        return Action(ActionType.PASS)


class MCTSBattleAgent(Agent):
    def __init__(self, agents=(RandomBattleAgent(), RandomBattleAgent())):
        self.agents = agents

    def seed(self, seed):
        for agent in self.agents:
            agent.seed(seed)

    def reset(self):
        pass

    def act(self, state, time_limit_ms=200, multiple=False):
        searcher = MCTS(agents=self.agents)

        if len(state.available_actions) == 1:
            return state.available_actions[0]

        start_time = int(time.time() * 1000.0)

        while True:
            current_time = int(time.time() * 1000.0)

            if current_time - start_time > time_limit_ms:
                break

            searcher.do_rollout(state)

        if multiple:
            action = searcher.choose_until_pass(state)
        else:
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


class NativeAgent(Agent):
    action_buffer = []

    def __init__(self, cmd, stateful=True, verbose=False):
        self.cmd = cmd
        self.stateful = stateful
        self.verbose = verbose
        self.initialized = False
        self.action_buffer = []

        self.raw_actions = ""

        self._process = None

    def initialize(self):
        self._process = pexpect.spawn(self.cmd, echo=False, encoding='utf-8')
        self.initialized = True

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.initialized:
            self._process.terminate()

            self._process = None
            self.initialized = False

    def seed(self, seed):
        pass

    def reset(self):
        self.action_buffer = []

        if self.initialized and self.stateful:
            self._process.terminate()

            self._process = None
            self.initialized = False

    @staticmethod
    def decode_actions(actions):
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

                if target == -1:
                    target = None

                decoded_actions.append(Action(ActionType.USE, origin, target))
            elif tokens[0] == 'SUMMON':
                origin = int(tokens[1])
                target = Lane(int(tokens[2]))

                decoded_actions.append(Action(ActionType.SUMMON, origin, target))
            elif tokens[0] == 'ATTACK':
                origin = int(tokens[1])
                target = int(tokens[2])

                if target == -1:
                    target = None

                decoded_actions.append(Action(ActionType.ATTACK, origin, target))

        return decoded_actions

    def act(self, state, multiple=False):
        if not self.initialized:
            self.initialize()

        return self._act(state, multiple)

    def _act(self, state, multiple=False):
        if self.action_buffer:
            if multiple:
                return list(reversed(self.action_buffer))
            else:
                return self.action_buffer.pop()

        self._process.write(str(state))

        actions = []

        try:
            raw_output = self._process.read_nonblocking(size=2048, timeout=2)

            self.raw_actions = raw_output.strip()

            actions = self.decode_actions(raw_output)

            if self.verbose:
                eprint(raw_output, end="")
        except TIMEOUT:
            print("WARNING: timeout")
        except EOF:
            print("WARNING: eof")

        if not actions:
            actions = [Action(ActionType.PASS)]

        if actions[-1].type != ActionType.PASS and state.is_battle():
            actions += [Action(ActionType.PASS)]

        if multiple:
            return actions
        else:
            self.action_buffer = list(reversed(actions))

            return self.action_buffer.pop()


class NativeBattleAgent(NativeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fake_draft(self, state):
        fake_state = State()

        play_first = state.current_player.id == 0
        deck = state.current_player.deck + state.current_player.hand

        if not play_first:
            fake_state.act(Action(ActionType.PASS))

        for turn in range(state.n):
            chosen_card = deck[turn]

            fake_state.current_player.hand = [chosen_card] * state.k

            self._process.write(str(fake_state))

            try:
                raw_output = self._process.read_nonblocking(size=2048, timeout=2)

                if self.verbose:
                    eprint(raw_output, end="")
            except TIMEOUT:
                print("WARNING: timeout")
            except EOF:
                print("WARNING: eof")

            fake_state.act(Action(ActionType.PASS))

        try:
            raw_output = self._process.read_nonblocking(size=2048, timeout=.1)

            if self.verbose:
                eprint(raw_output, end="")
        except TIMEOUT:
            pass
        except EOF:
            pass

    def act(self, state, multiple=False):
        if not self.initialized:
            self.initialize()
            self.fake_draft(state)

        return super()._act(state, multiple)


class NativeDraftAgent(NativeAgent):
    pass


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


class MaxAttackDraftAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

    def act(self, state):
        hand = state.current_player.hand
        index, max_attack = 0, 0

        for i in range(3):
            if isinstance(hand[i], Creature) and hand[i].attack > max_attack:
                index, max_attack = i, hand[i].attack

        return Action(ActionType.PICK, index)


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
        -666, 65, 50, 80, 50, 70, 71, 115, 71, 73,
        43, 77, 62, 63, 50, 66, 60, 66, 90, 75,
        50, 68, 67, 100, 42, 63, 67, 52, 69, 90,
        60, 47, 87, 81, 67, 62, 75, 94, 56, 62,
        51, 61, 43, 54, 97, 64, 67, 49, 109, 111,
        89, 114, 93, 92, 89, 2, 54, 25, 63, 76,
        58, 99, 79, 19, 82, 115, 106, 104, 146, 98,
        70, 56, 65, 52, 54, 65, 55, 77, 48, 84,
        115, 75, 89, 68, 80, 71, 46, 73, 69, 47,
        63, 70, 11, 71, 54, 85, 77, 77, 64, 82,
        62, 49, 43, 78, 67, 72, 67, 36, 48, 75,
        -8, 82, 69, 32, 87, 98, 124, 35, 60, 59,
        49, 72, 54, 35, 22, 50, 54, 51, 54, 59,
        38, 31, 43, 62, 55, 57, 41, 70, 38, 76,
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


class UJI1DraftAgent(Agent):
    def __init__(self):
        self.picked = [0] * 10
        self.preference = [5, 5, 3, 3, 3, 4, 4, 3, 0, 0]

    def seed(self, seed):
        random.seed(seed)

    def reset(self):
        self.picked = [0] * 10

    @staticmethod
    def get_index(card):
        if isinstance(card, Creature):
            return min(6, max(0, card.cost - 1))
        elif isinstance(card, GreenItem):
            return 7
        elif isinstance(card, RedItem):
            return 8
        elif isinstance(card, BlueItem):
            return 9
        else:
            raise ValueError

    def act(self, state):
        weights = []
        cards = state.current_player.hand
        indexes = list(map(self.get_index, cards))

        for index, card in zip(indexes, cards):
            if card.id == 151:
                p = 100
            elif isinstance(card, GreenItem) or isinstance(card, BlueItem):
                p = -100
            else:
                p = self.preference[index] - self.picked[index]

                if isinstance(card, Creature) and card.has_ability('G'):
                    p += 6

            weights.append(p)

        chosen_card = int(np.argmax(weights))

        self.picked[indexes[chosen_card]] += 1

        return Action(ActionType.PICK, chosen_card)


class UJI2DraftAgent(Agent):
    def __init__(self):
        self.picked = [0] * 10
        self.preference = [5, 4, 4, 3, 3, 3, 2, 2, 2, 2]

    def seed(self, seed):
        random.seed(seed)

    def reset(self):
        self.picked = [0] * 10

    @staticmethod
    def get_index(card):
        if isinstance(card, Creature):
            return min(6, max(0, card.cost - 1))
        elif isinstance(card, GreenItem):
            return 7
        elif isinstance(card, RedItem):
            return 8
        elif isinstance(card, BlueItem):
            return 9
        else:
            raise ValueError

    def act(self, state):
        weights = []
        cards = state.current_player.hand
        indexes = list(map(self.get_index, cards))

        for index, card in zip(indexes, cards):
            p = self.preference[index] - self.picked[index]

            if isinstance(card, Creature) and card.has_ability('G'):
                p += 6

            weights.append(max(0, p))

        if sum(weights) == 0:
            chosen_card = random.randint(0, 2)
        else:
            chosen_card = random.choices(range(3), weights)[0]

        self.picked[indexes[chosen_card]] += 1

        return Action(ActionType.PICK, chosen_card)


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

    def _coac_eval(self, key):
        return lambda card: self.scores[key].index(card.id)

    def act(self, state):
        key = 'p1' if state.current_player.id == PlayerOrder.FIRST else 'p2'

        if self.drafted > 0 and self.creatures_drafted / self.drafted < 0.4:
            key += '_creature'

        chosen_card = np.argmin(list(map(self._coac_eval(key), state.current_player.hand)))

        self.drafted += 1

        if isinstance(state.current_player.hand[chosen_card], Creature):
            self.creatures_drafted += 1

        return Action(ActionType.PICK, chosen_card)


class Coac2DraftAgent(CoacDraftAgent):
    new_scores = {
        'p1': [
            68, 7, 116, 65, 116, 151, 69, 48, 53, 51,
            44, 67, 29, 18, 84, 18, 158, 28, 64, 80,
            33, 85, 32, 147, 103, 37, 54, 52, 50, 99,
            23, 87, 66, 81, 148, 88, 150, 121, 82, 95,
            115, 133, 152, 19, 109, 157, 105, 3, 75, 96,
            114, 9, 106, 144, 129, 128, 17, 128, 12, 11,
            145, 15, 21, 8, 134, 155, 141, 70, 90, 135,
            104, 41, 112, 61, 5, 97, 73, 26, 73, 6, 36,
            86, 77, 83, 13, 93, 93, 93, 149, 59, 159, 74,
            94, 38, 98, 126, 39, 30, 137, 100, 22, 62,
            118, 22, 118, 1, 47, 71, 4, 91, 27, 56, 119,
            101, 45, 16, 146, 58, 120, 142, 127, 25, 108,
            132, 40, 14, 76, 125, 102, 131, 123, 2, 35,
            130, 107, 43, 63, 31, 138, 124, 154, 78, 46,
            24, 10, 136, 113, 60, 57, 92, 55, 117, 55,
            153, 20, 156, 143, 110, 160, 140
        ],
        'p2': [
            68, 7, 65, 49, 116, 69, 51, 151, 53, 51,
            44, 67, 29, 139, 28, 84, 28, 158, 64, 80,
            33, 85, 32, 147, 103, 37, 50, 54, 50, 99,
            23, 87, 66, 81, 148, 88, 150, 121, 82, 95,
            115, 133, 152, 19, 109, 157, 105, 3, 96,
            75, 114, 9, 106, 144, 129, 17, 111, 128,
            12, 15, 11, 145, 21, 8, 134, 155, 141, 70,
            90, 135, 104, 112, 41, 61, 5, 97, 26, 34,
            73, 86, 6, 86, 83, 77, 89, 13, 79, 93, 59,
            149, 159, 74, 94, 38, 126, 98, 39, 30, 100,
            62, 137, 122, 22, 72, 118, 1, 47, 71, 4, 91,
            56, 27, 119, 101, 45, 146, 16, 120, 58, 142,
            25, 127, 108, 132, 40, 14, 76, 125, 102, 123,
            131, 2, 35, 130, 107, 43, 63, 31, 138, 124,
            154, 78, 46, 24, 10, 136, 113, 60, 57, 92,
            117, 55, 42, 153, 20, 156, 143, 110, 160, 140
        ],
        'p1_creature': [
            68, 7, 65, 49, 116, 69, 151, 48, 53, 51,
            44, 67, 29, 139, 84, 18, 158, 28, 64, 80,
            33, 85, 32, 147, 103, 37, 54, 52, 50, 99,
            23, 87, 66, 81, 148, 88, 82, 150, 82, 95,
            115, 133, 152, 19, 109, 157, 105, 3, 75,
            96, 114, 9, 106, 144, 129, 17, 111, 128,
            12, 11, 145, 15, 21, 8, 134, 155, 141, 70,
            90, 135, 104, 41, 112, 61, 5, 97, 26, 34,
            73, 6, 36, 86, 77, 83, 13, 89, 79, 93, 149,
            59, 159, 74, 94, 38, 98, 126, 39, 30, 137,
            100, 62, 122, 22, 72, 118, 1, 47, 71, 4, 91,
            27, 56, 119, 101, 45, 16, 146, 58, 120, 142,
            127, 25, 108, 132, 40, 14, 76, 131, 125, 131,
            123, 2, 35, 130, 107, 43, 63, 31, 138, 124,
            154, 78, 46, 24, 10, 136, 113, 60, 57, 92,
            117, 42, 55, 156, 153, 156, 143, 110, 160, 140
        ],
        'p2_creature': [
            68, 7, 116, 65, 116, 69, 151, 48, 53, 51,
            44, 67, 29, 139, 84, 18, 158, 28, 64, 80,
            33, 85, 32, 147, 103, 37, 54, 52, 50, 99,
            23, 87, 66, 81, 148, 88, 150, 121, 82, 95,
            115, 133, 152, 19, 109, 157, 105, 3, 75,
            96, 114, 9, 106, 144, 129, 17, 111, 128,
            12, 15, 11, 145, 21, 8, 134, 155, 141, 70,
            90, 135, 104, 112, 41, 61, 5, 97, 26, 34,
            73, 6, 36, 86, 77, 83, 89, 13, 79, 93, 59,
            149, 159, 74, 94, 38, 126, 98, 39, 30, 100,
            62, 137, 122, 22, 72, 118, 1, 47, 71, 4, 91,
            56, 27, 119, 101, 45, 146, 16, 120, 58, 142,
            25, 127, 108, 132, 40, 14, 76, 125, 102, 123,
            131, 2, 35, 130, 107, 43, 63, 31, 138, 124,
            154, 78, 46, 24, 10, 136, 113, 60, 57, 92,
            117, 55, 42, 153, 20, 156, 143, 110, 160, 140
        ]
    }

    def _coac_eval(self, key):
        def _coac_card_eval(card):
            try:
                return self.new_scores[key].index(card.id)
            except ValueError:
                return float("inf")

        return _coac_card_eval


class ChadDraftAgent(Agent):
    scores_p1 = [
        -177, -133, 130, 99, 105, 190, 164, 170, -15, -132, 186, -47, 41, 108, -1, 17, -121,
        112, 1, -9, -177, 10, -21, -151, -108, 188, 171, 63, 37, 170, 21, -129, 33, -3, -97,
        -30, -138, 120, 128, 166, 200, -13, -131, -197, 128, -85, 43, 42, -82, -71, 89, 134,
        170, 89, 17, 56, -188, 103, 27, -69, -129, -9, 36, 133, -28, 169, -114, 196, 167, 26,
        -51, 4, -80, 16, 167, 11, -182, 98, -147, 41, 11, 32, -143, 180, 104, 168, 19, -31,
        -72, 64, -183, -77, 185, 95, -47, -40, 172, 28, 93, -147, 111, -17, 93, -95, -18, -3,
        -14, -99, -171, 116, 38, -184, 61, -1, -95, -82, 60, -57, 108, -194, -16, -30, 168,
        122, 42, 117, 149, 97, 21, -12, 127, 83, 161, -61, -23, 144, 62, -26, -56, -199, -36,
        -181, -77, 187, 69, 17, -150, -83, 24, -110, -158, -149, -139, -121, -96, 148, -64,
        -111, 108, -192
    ]
    scores_p2 = [
        175, 128, -42, 107, 137, 180, 181, -125, 110, 138, 190, -51, 46, -135, -70, 37, -121,
        118, 2, 45, 106, 1, -29, 125, 64, 179, 166, -98, 24, 167, 21, -117, -177, -17, 20, 161,
        -165, 157, 149, 166, 192, -108, -131, -197, 6, -55, 63, 42, 116, -165, 19, -129, 80,
        37, -157, 56, -114, 151, -86, -62, -116, -117, 61, 130, -146, -81, -127, -200, 167, 38,
        -46, -162, 8, -11, -97, 41, -176, 52, -94, 185, 103, 96, -152, 199, 112, -177, 21,
        -104, 97, 77, -179, -77, 176, -199, -188, 130, 172, 8, -101, 156, -24, 199, 97, -17,
        -43, -54, -152, -99, -181, -89, 151, -74, -199, -67, -157, -57, 117, 157, 180, -194,
        -28, 159, 91, 38, -156, 117, -181, 151, -6, -12, 171, 89, -97, 181, 28, -148, -43, -18,
        -69, 190, 84, 196, -76, -127, -83, -29, 118, -17, 120, 129, 166, -134, -17, -121, -94,
        179, -64, 110, 115, -184
    ]

    def seed(self, seed):
        pass

    def reset(self):
        pass

    def _chad_eval_p1(self, card):
        return self.scores_p1[card.id - 1]

    def _chad_eval_p2(self, card):
        return self.scores_p2[card.id - 1]

    def act(self, state):
        if state.current_player.id == PlayerOrder.FIRST:
            card_eval_func = self._chad_eval_p1
        else:
            card_eval_func = self._chad_eval_p2

        index = np.argmax(list(map(card_eval_func, state.current_player.hand)))

        return Action(ActionType.PICK, index)


class HistorylessDraftAgent(Agent):
    scores = [
        0.4246, 0.3888, 0.4012, 0.165, 0.3894, 0.3404, 0.4158, 0.2842, 0.179,
        0.4244, 0.2998, 0.125, 0.176, 0.303, 0.0966, 0.24, 0.1032, 0.161,
        0.0552, 0.183, 0.072, 0.0518, 0.016, 0.4282, 0.3806, 0.3204, 0.3396,
        0.2084, 0.2948, 0.2612, 0.3138, 0.1708, 0.098, 0.0344, 0.1234, 0.0174,
        0.0122, 0.4432, 0.5146, 0.329, 0.3376, 0.351, 0.128, 0.075, 0.1922,
        0.0264, 0.3178, 0.4438, 0.3614, 0.3162, 0.1128, 0.143, 0.1598, 0.278,
        0.1786, 0.0418, 0.0256, 0.0756, 0.0122, 0.009, 0.0036, 0.0764, 0.154,
        0.2744, 0.4402, 0.3234, 0.108, 0.1368, 0.3172, 0.2906, 0.202, 0.2776,
        0.277, 0.3182, 0.1668, 0.2166, 0.0506, 0.043, 0.0282, 0.1418, 0.018,
        0.2938, 0.4038, 0.4532, 0.142, 0.0744, 0.1258, 0.069, 0.1532, 0.0138,
        0.417, 0.2948, 0.4266, 0.2422, 0.2904, 0.3762, 0.32, 0.2312, 0.2228,
        0.1458, 0.216, 0.2636, 0.174, 0.2694, 0.1756, 0.2398, 0.1414, 0.1288,
        0.0532, 0.036, 0.197, 0.1268, 0.0604, 0.1668, 0.1122, 0.109, 0.2422,
        0.1332, 0.1366, 0.1612, 0.038, 0.086, 0.2028, 0.1736, 0.0386, 0.0596,
        0.014, 0.0634, 0.0236, 0.004, 0.0992, 0.0624, 0.0936, 0.0268, 0.0192,
        0.2328, 0.1426, 0.0732, 0.0688, 0.0642, 0.447, 0.209, 0.1602, 0.4706,
        0.3118, 0.2184, 0.2874, 0.1916, 0.088, 0.4618, 0.394, 0.2502, 0.0038,
        0.0034, 0.0096, 0.0024, 0.0016, 0.0178, 0.0056, 0.0046
    ]

    def seed(self, seed):
        pass

    def reset(self):
        pass

    def _card_eval(self, card):
        return self.scores[card.id - 1]

    def act(self, state):
        index = np.argmax(list(map(self._card_eval, state.current_player.hand)))

        return Action(ActionType.PICK, index)


class RLDraftAgent(Agent):
    def __init__(self, model):
        self.model = model

        self.hidden_states = None
        self.dones = None

    def seed(self, seed):
        pass

    def reset(self):
        self.hidden_states = None
        self.dones = None

    def act(self, state):
        action, self.hidden_states = \
            self.model.predict(state, state=self.hidden_states,
                               mask=self.dones, deterministic=True)

        return action


class RLBattleAgent(Agent):
    def __init__(self, model):
        self.model = model

        self.hidden_states = None
        self.dones = None

    def seed(self, seed):
        pass

    def reset(self):
        self.hidden_states = None
        self.dones = None

    def act(self, state, action_masks):
        action, self.hidden_states = \
            self.model.predict(state, state=self.hidden_states,
                               deterministic=True, action_masks=action_masks)

        return action


class TabularRLDraftAgent(Agent):

    def __init__(self, policy_path='policy-3M.csv', has_header=True):
        with open(policy_path, 'r') as policy_file:
            lines = policy_file.readlines()

        self.policy = {None: 0}

        for line in lines[1 if has_header else 0:]:
            c1, c2, c3, a = map(int, line.split(';'))

            self.policy[(c1, c2, c3)] = a

    def seed(self, seed):
        pass

    def reset(self):
        pass

    def act(self, state):
        cards = tuple(sorted(map(lambda c_id: c_id - 1, map(attrgetter('id'), state.current_player.hand))))

        return Action(ActionType.PICK, self.policy[cards])


draft_agents = {
    "pass": PassDraftAgent,
    "random": RandomDraftAgent,
    "rule-based": RuleBasedDraftAgent,
    "max-attack": MaxAttackDraftAgent,
    "icebox": IceboxDraftAgent,
    "closet-ai": ClosetAIDraftAgent,
    "uji1": UJI1DraftAgent,
    "uji2": UJI2DraftAgent,
    "coac": CoacDraftAgent,
    "coac2": Coac2DraftAgent,
    "chad": ChadDraftAgent,
    "historyless": HistorylessDraftAgent,
    "rl": RLDraftAgent,
    'tabular-rl': TabularRLDraftAgent
}

battle_agents = {
    "pass": PassBattleAgent,
    "random": RandomBattleAgent,
    "greedy": GreedyBattleAgent,
    "rule-based": RuleBasedBattleAgent,
    "max-attack": MaxAttackBattleAgent,
    "coac": CoacBattleAgent,
    "mcts": MCTSBattleAgent
}


def parse_draft_agent(agent_name: str) -> Type:
    return draft_agents[agent_name.lower().replace(" ", "-")]


def parse_battle_agent(agent_name: str) -> Type:
    return battle_agents[agent_name.lower().replace(" ", "-")]
