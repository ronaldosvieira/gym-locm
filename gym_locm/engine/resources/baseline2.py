class Card():
    def __init__(self, instance_id, location, type, attack, defense, keywords):
        self.instance_id = instance_id
        self.location = location
        self.type = type
        self.attack = attack
        self.defense = defense
        self.keywords = keywords


while True:
    health, mana, deck, rune, draw = [int(j) for j in input().split()]
    input()
    opponent_hand, opponent_actions = [int(i) for i in input().split()]
    for i in range(opponent_actions):
        card_number_and_action = input()
    card_count = int(input())

    myonboard = []
    myinhand = []
    oppguards = []
    for i in range(card_count):
        card_name, instance_id, location, type, cost, attack, defense, keywords, my_health, opponent_health, card_draw, lane = input().split()
        card = Card(int(instance_id), int(location), int(type), int(attack), int(defense), keywords)
        if card.location == 1: myonboard.append(card)
        if card.location == 0: myinhand.append(card)
        if card.location == -1 and 'G' in keywords: oppguards.append(card)

    if mana == 0:
        pick = 0
        att = -1
        for n in (0, 1, 2):
            if myinhand[n].type == 0 and myinhand[n].attack > att:
                pick = n
                att = myinhand[n].attack
        print('PICK', pick)
    else:
        myonboard.sort(key=lambda c: c.attack, reverse=True)
        myinhand.sort(key=lambda c: c.attack, reverse=True)
        oppguards.sort(key=lambda c: c.defense, reverse=True)
        actions = []

        for c in myonboard:
            actions.append('ATTACK ' + str(c.instance_id) + ' -1')
            for t in oppguards:
                actions.append('ATTACK ' + str(c.instance_id) + ' ' + str(t.instance_id))
        for c in myinhand:
            if c.type == 0:
                actions.append('SUMMON ' + str(c.instance_id) + ' 0')
                actions.append('SUMMON ' + str(c.instance_id) + ' 1')

        if actions:
            print(';'.join(actions))
        else:
            print("PASS")