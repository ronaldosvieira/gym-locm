# gym-locm

A collection of [OpenAI Gym](https://github.com/openai/gym) environments for the collectible card game [Legends of Code and Magic (LOCM)](https://jakubkowalski.tech/Projects/LOCM/).

## Installation
```
git clone https://github.com/ronaldosvieira/gym-locm.git
cd gym-locm
pip install -e .
```

## Usage

```python
import gym
import gym_locm

env = gym.make('LOCM-XXX-vX')

done = False
while not done:
    action = ...  # Your agent code here
    obs, reward, done, _ = env.step(action)
    env.render()
```

## Environments

A match of LOCM has two phases: the **draft**, where the players build their decks, and the **battle**, where the playing actually occurs.

A reward of *1* is given if the controlled player wins the battle phase, and *-1* otherwise. There are no draws. 

### Draft phase only
 ```python
env = gym.make("LOCM-draft-v0")
```

The draft phase is played. A default (configurable) policy is used in the battle phase.

**State**: a 3 x 16 matrix (16 features from each of the 3 cards). 

**Actions**: 0-2 (chooses first, second or third card).

### Battle phase only
 ```python
env = gym.make("LOCM-battle-v0")
```

The battle phase is played. A default (configurable) policy is used in the draft phase.

**State**: a vector with 3 x 20 + 8 values (16 features from each of the 20 possible cards plus 4 features from each player).

**Actions**: 0-96 (pass, summon, use and attack with all possible origins and targets).
<details>
  <summary>Click to see all actions</summary>
    
     0: PASS
     1: SUMMON (card at index 0 of player's hand) 0
     2: SUMMON (card at index 0 of player's hand) 1
     3: SUMMON (card at index 1 of player's hand) 0
     4: SUMMON (card at index 1 of player's hand) 1
     5: SUMMON (card at index 2 of player's hand) 0
                          ⋮
    16: SUMMON (card at index 7 of player's hand) 1
    17: USE (card at index 0 of player's hand) -1
    18: USE (card at index 0 of player's hand) (1st creature at lane 0)
    19: USE (card at index 0 of player's hand) (2nd creature at lane 0)
    20: USE (card at index 0 of player's hand) (3rd creature at lane 0)
    21: USE (card at index 0 of player's hand) (1st creature at lane 1)
    22: USE (card at index 0 of player's hand) (2nd creature at lane 1)
    23: USE (card at index 0 of player's hand) (3rd creature at lane 1)
    24: USE (card at index 1 of player's hand) -1
    25: USE (card at index 1 of player's hand) (1st creature at lane 0)
                          ⋮
    72: USE (card at index 7 of player's hand) (3rd creature at lane 1)
    73: ATTACK (1st creature at player's lane 0) -1
    74: ATTACK (1st creature at player's lane 0) (1st creature at opponent's lane 0)
    75: ATTACK (1st creature at player's lane 0) (2nd creature at opponent's lane 0)
    76: ATTACK (1st creature at player's lane 0) (3rd creature at opponent's lane 0)
    77: ATTACK (2nd creature at player's lane 0) -1
    78: ATTACK (2nd creature at player's lane 0) (1st creature at opponent's lane 0)
                          ⋮
    85: ATTACK (1st creature at player's lane 1) -1
                          ⋮
    96: ATTACK (3rd creature at player's lane 1) (3rd creature at opponent's lane 0)
    
</details>

### Full match
```python
env = gym.make("LOCM-v0")
```

A full match is played. The draft phase happens in the first 30 turns, with the battle phase taking place on the subsequent turns.

States and actions are the same as listed above, changing according to the current phase.

### Two-player variations
 ```python
env = gym.make("LOCM-draft-2p-v0")
env = gym.make("LOCM-battle-2p-v0")
env = gym.make("LOCM-2p-v0")
```

Both players are controlled alternately. A reward of *1* is given if the first player wins, and *-1* otherwise. 

## Citing

If you use gym-locm in your work, please consider citing it with:

```
@misc{gym-locm,
    author = {Vieira, Ronaldo; Chaimowicz, Luiz; Tavares, Anderson Rocha},
    title = {OpenAI Gym Environments for Legends of Code and Magic},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ronaldosvieira/gym-locm}},
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
