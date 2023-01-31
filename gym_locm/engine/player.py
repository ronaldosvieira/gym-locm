from gym_locm.exceptions import EmptyDeckError, FullHandError


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
