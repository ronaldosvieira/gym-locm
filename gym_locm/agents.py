from gym_locm.engine import *
from gym_locm.helpers import *


def parse_action(state, action_type, origin, target):
    current_player = state.players[state.current_player]
    opposing_player = state.players[state.current_player.opposing()]

    if action_type == BattleActionType.SUMMON:
        origin = current_player.hand[origin]
        target = Lane.LEFT if target == 0 else Lane.RIGHT
    elif action_type == BattleActionType.ATTACK:
        lane = Lane.LEFT if origin < 3 else Lane.RIGHT
        creature = origin if lane == Lane.LEFT else origin - 3

        origin = current_player.lanes[lane][creature]

        lane = Lane.LEFT if target < 3 else Lane.RIGHT
        creature = target if lane == Lane.LEFT else target - 3

        target = None if target == 6 else opposing_player.lanes[lane][creature]
    elif action_type == BattleActionType.USE:
        origin = current_player.hand[origin]

        player = current_player if target < 6 else opposing_player
        lane = Lane.LEFT if (target % 6) < 3 else Lane.RIGHT
        creature = (target % 6) if lane == Lane.LEFT else (target % 6) - 3

        target = None if target == 12 else player.lanes[lane][creature]

    return BattleAction(action_type, origin, target)


class BattleAgent:
    def act(self, state):
        pass


class PassBattleAgent(BattleAgent):
    def act(self, state):
        return BattleAction(BattleActionType.PASS)


class RandomBattleAgent(BattleAgent):
    def act(self, state):
        available_actions = state.available_actions()

        available_actions[BattleActionType.PASS] = False

        if not any(available_actions.values()):
            return BattleAction(BattleActionType.PASS)

        probabilities = np.array([1 if action else 0 for action in available_actions.values()])
        probabilities[-1] = 0

        probabilities = probabilities / sum(probabilities)
        action_type = np.random.choice(BattleActionType, p=probabilities)

        origin, target = None, None

        available_origins = available_actions[action_type]

        probabilities = np.array([1 if origin else 0 for origin in available_origins])
        probabilities = probabilities / sum(probabilities)
        origin = np.random.choice(range(len(available_origins)), p=probabilities)

        available_targets = available_origins[origin]

        probabilities = np.array([1 if target else 0 for target in available_targets])
        probabilities = probabilities / sum(probabilities)
        target = np.random.choice(range(len(available_targets)), p=probabilities)

        return parse_action(state, action_type, origin, target)


class RuleBasedBattleAgent(BattleAgent):
    def __init__(self):
        self.last_action = None

    def act(self, state):
        current_player = state.players[state.current_player]
        opposing_player = state.players[state.current_player.opposing()]

        castable = list(filter(has_enough_mana(current_player.mana),
                               current_player.hand))
        summonable = list(filter(is_it(Creature), castable))

        creatures = [c for lane in current_player.lanes for c in lane]
        opp_creatures = [c for lane in opposing_player.lanes for c in lane]
        can_attack = list(filter(Creature.able_to_attack, creatures))

        green_items = list(filter(is_it(GreenItem), castable))
        red_items = list(filter(is_it(RedItem), castable))
        blue_items = list(filter(is_it(BlueItem), castable))

        if summonable:
            creature = np.random.choice(summonable)
            lane = Lane.LEFT if np.random.choice([0, 1]) == 0 else Lane.RIGHT

            action = BattleAction(BattleActionType.SUMMON,
                                  creature,
                                  lane)

            if self.last_action != action:
                self.last_action = action
                return action

        if can_attack:
            creature = np.random.choice(can_attack)
            lane = Lane.LEFT if creature in current_player.lanes[0] else Lane.RIGHT

            opp_creatures = opposing_player.lanes[lane]
            guards = list(filter(lambda c: c.has_ability('G'), opp_creatures))

            action = BattleAction(BattleActionType.ATTACK,
                                  creature,
                                  np.random.choice(guards) if guards else None)

            if self.last_action != action:
                self.last_action = action
                return action

        if creatures and green_items:
            action = BattleAction(BattleActionType.USE,
                                  np.random.choice(green_items),
                                  np.random.choice(creatures))

            if self.last_action != action:
                self.last_action = action
                return action

        if opp_creatures and red_items:
            action = BattleAction(BattleActionType.USE,
                                  np.random.choice(red_items),
                                  np.random.choice(opp_creatures))

            if self.last_action != action:
                self.last_action = action
                return action

        if blue_items:
            action = BattleAction(BattleActionType.USE,
                                  np.random.choice(blue_items))

            if self.last_action != action:
                self.last_action = action
                return action

        action = BattleAction(BattleActionType.PASS)

        self.last_action = action

        return action
