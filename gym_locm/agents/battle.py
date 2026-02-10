import random
from operator import attrgetter
from gym_locm.agents import Agent
from gym_locm.engine import (
    Action,
    ActionType,
    Lane,
    Creature,
    GreenItem,
    RedItem,
    BlueItem,
)
from gym_locm.util import is_it, has_enough_mana


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

            if (
                isinstance(card, Creature)
                and card.cost <= state.current_player.mana
                and len(state.current_player.lanes[current_lane]) < 3
            ):
                action = Action(ActionType.SUMMON, origin, current_lane)

                return action

            elif (
                isinstance(card, GreenItem)
                and card.cost <= state.current_player.mana
                and friends
            ):
                target = friends[0].instance_id

                return Action(ActionType.USE, origin, target)
            elif (
                isinstance(card, RedItem)
                and card.cost <= state.current_player.mana
                and foes
            ):
                target = foes[0].instance_id

                return Action(ActionType.USE, origin, target)
            elif isinstance(card, BlueItem) and card.cost <= state.current_player.mana:
                return Action(ActionType.USE, origin, None)

        for card in state.current_player.lanes[Lane.LEFT]:
            origin = card.instance_id

            if card.can_attack and not card.has_attacked_this_turn:
                for enemy in state.opposing_player.lanes[Lane.LEFT]:
                    if enemy.has_ability("G"):
                        target = enemy.instance_id

                        return Action(ActionType.ATTACK, origin, target)

                return Action(ActionType.ATTACK, origin, None)

        for card in state.current_player.lanes[Lane.RIGHT]:
            origin = card.instance_id

            if card.can_attack and not card.has_attacked_this_turn:
                for enemy in state.opposing_player.lanes[Lane.RIGHT]:
                    if enemy.has_ability("G"):
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
        lanes = zip(list(Lane), state.current_player.lanes, state.opposing_player.lanes)

        for lane, friends, foes in lanes:
            guard_foes = filter(lambda c: c.has_ability("G"), foes)

            friends = filter(Creature.able_to_attack, friends)
            friends = sorted(friends, key=attrgetter("attack"), reverse=True)

            for creature in friends:
                try:
                    target = next(guard_foes)
                except StopIteration:
                    target = None

                return Action(ActionType.ATTACK, creature.instance_id, target)

        creatures_in_hand = filter(is_it(Creature), state.current_player.hand)
        creatures_in_hand = filter(
            has_enough_mana(state.current_player.mana), creatures_in_hand
        )
        creatures_in_hand = sorted(
            creatures_in_hand, key=attrgetter("attack"), reverse=True
        )

        lanes = (l for l in Lane if len(state.current_player.lanes[l]) < 3)

        try:
            for creature in creatures_in_hand:
                return Action(ActionType.SUMMON, creature.instance_id, next(lanes))
        except StopIteration:
            pass

        return Action(ActionType.PASS)
