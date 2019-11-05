from operator import attrgetter

import mplcursors
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from gym_locm.engine import load_cards, Creature, GreenItem, RedItem, BlueItem


card_types = {Creature: 0, GreenItem: 1, RedItem: 2, BlueItem: 3}


def encode_card(card):
    card_type = card_types[type(card)]
    cost = card.cost
    attack = card.attack
    defense = max(card.defense, -12)
    keywords = list(map(int, map(card.keywords.__contains__, 'BCDGLW')))
    player_hp = card.player_hp
    enemy_hp = card.enemy_hp
    card_draw = card.card_draw

    return [card_type, cost, attack, defense, player_hp,
            enemy_hp, card_draw] + keywords


cards = load_cards()

encoded_cards = map(encode_card, cards)

columns = ['type', 'cost', 'attack', 'defense', 'player_hp',
           'enemy_hp', 'card_draw', 'B', 'C', 'D', 'G', 'L', 'W']

df = pd.DataFrame(data=encoded_cards, columns=columns)

# print(df)  # see cards data frame

# get additional columns
names = pd.DataFrame(data=map(attrgetter('name'), cards), columns=['name'])
costs = df['cost']

types_list = ['Creature', 'Green item', 'Red item', 'Blue item']
colors_list = ['y', 'g', 'r', 'b']

types = list(map(types_list.__getitem__, df['type']))
types = pd.DataFrame(data=types, columns=['type'])

type_to_color = dict(zip(types_list, colors_list))

colors = types.applymap(type_to_color.__getitem__)
colors.rename(columns={'type': 'color'}, inplace=True)

# select features to be used
features = columns[1:]
df = df.loc[:, features].values

# standardize features data
df = pd.DataFrame(data=StandardScaler().fit_transform(df), columns=features)

# print(df)  # see standardized cards

# apply PCA for two components
pca = PCA(n_components=2)
df = pd.DataFrame(data=pca.fit_transform(df), columns=['pc1', 'pc2'])

# join the 2D coordinates to the metadata
df = pd.concat([names, costs, types, colors, df], axis=1)

# print(df)  # see cards in two dimensions

# print amount of information represented by pcs
# print(sum(pca.explained_variance_ratio_))

# plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('LOCM cards in two dimensions', fontsize=20)

targets = ['Creature', 'Green item', 'Red item', 'Blue item']

scatter = ax.scatter(df['pc1'], df['pc2'], c=df['color'], s=10 + 10 * df['cost'],
                     edgecolors='k', linewidth=0.5)

for name, _, _, _, x, y in df.values:
    annot = ax.annotate(name, (x, y), (x - 0.075 * len(name), y + 0.15))
    annot.set_visible(False)

lp = lambda t: plt.plot([], color=type_to_color[t], ms=8, mec="none",
                        label=t, ls="", marker="o", linewidth=0.5)[0]

ax.legend(handles=[lp(i) for i in types_list])
ax.grid()

cursor = mplcursors.cursor(hover=False)
cursor.connect(
    "add", lambda sel: sel.annotation.set_text(df["name"][sel.target.index]))

plt.show()
