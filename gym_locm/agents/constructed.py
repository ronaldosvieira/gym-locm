import random
from gym_locm.agents import Agent
from gym_locm.engine import Action, ActionType, Card, Creature, GreenItem


class MaxAttackConstructedAgent(Agent):
    def seed(self, seed):
        pass

    def reset(self):
        pass

    def act(self, state):
        hand = state.current_player.hand
        index, max_attack = 0, 0

        for i, (card, mask) in enumerate(
            zip(hand, state.deck_building_phase.action_mask())
        ):
            if not mask:
                continue

            if isinstance(card, Creature) and card.attack > max_attack:
                index, max_attack = i, card.attack

        return Action(ActionType.CHOOSE, index)


class InspiraiConstructedAgent(Agent):
    area = -3.779216981947414
    cost = -3.7998646581933975
    att_def_sum = -4.450924576381176
    att_def_hm = 1.1914191568197516
    monster = 1.4308790882830391
    lethal = -3.983350714836453
    ward = 2.1011340087179864
    guard = 2.156171607641912
    breakthrough = -3.1048119973463457
    drain = 0.8286945260657275
    red_blue = -0.23560153858421984
    green = -0.8618647982692815
    green_ward = 1.1256445689216639
    monster_multi = -2.6645675460661264
    monster_no_att = -1.0590017361786237
    card_draw = 0.37833242963248814

    min_monster = 8
    twice = True

    def __init__(self):
        self.selected_card_ids = []

    def seed(self, seed):
        pass

    def reset(self):
        self.selected_card_ids = []

    def _eval_card(self, card: Card) -> float:
        card_lethal = int(card.has_ability("L"))
        card_ward = int(card.has_ability("W"))
        card_guard = int(card.has_ability("G"))
        card_breakthrough = int(card.has_ability("B"))
        card_drain = int(card.has_ability("D"))
        card_charge = int(card.has_ability("C"))

        score = 0.0
        area = self.area if card.area != 0 else 1.0
        score += self.cost * card.cost
        score += (
            self.att_def_sum
            * area
            * (abs(card.attack) + abs(card.defense))
            / max(1, card.cost)
        )

        if abs(card.attack) + abs(card.defense) > 0:
            att_def_hm = (
                abs(card.attack)
                * abs(card.defense)
                / (abs(card.attack) + abs(card.defense))
            )
            score += self.att_def_hm * area * att_def_hm

        if isinstance(card, Creature):
            monster_score = (
                self.monster
                + card_lethal * self.lethal
                + card_ward * self.ward
                + card_guard * self.guard
                + card_breakthrough * self.breakthrough
                + card_drain * self.drain
            )
            monster_score *= area
            score += monster_score

        elif isinstance(card, GreenItem):
            score += self.red_blue * (abs(card.attack) + abs(card.defense)) * area
        else:
            score += self.green + self.green_ward * card_ward * area

        if (
            isinstance(card, Creature)
            and card_charge
            and card_lethal
            and card.attack > 0
        ):
            score += self.monster_multi * area

        if isinstance(card, Creature) and card.attack == 0:
            score += self.monster_no_att

        score += self.card_draw * area * card.card_draw

        return score

    def _eval_state(self, state):
        cards = [c for c in state.current_player.hand]
        cards = sorted(cards, key=self._eval_card, reverse=True)

        selected_card_ids = {}
        min_monster = self.min_monster

        for card in cards:
            if min_monster <= 0:
                break

            if isinstance(card, Creature):
                selected_card_ids[card.id] = 1 + self.twice
                min_monster -= 1 + self.twice

        for card in cards:
            if sum(selected_card_ids.values()) >= 30:
                break

            if selected_card_ids.get(card.id, 0) >= 2:
                continue

            selected_card_ids[card.id] = selected_card_ids.get(card.id, 0) + (
                1 + self.twice
            )

        selected_card_ids = sum([[k] * v for k, v in selected_card_ids.items()], [])

        self.selected_card_ids = list(reversed(selected_card_ids[:30]))

    def act(self, state):
        if not self.selected_card_ids:
            self._eval_state(state)

        return Action(ActionType.CHOOSE, self.selected_card_ids.pop())
