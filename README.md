# gym-locm

A collection of [Gymnasium](https://gymnasium.farama.org/) environments for the collectible card 
game [Legends of Code and Magic (LOCM)](https://legendsofcodeandmagic.com/).

## Installation

Python 3.12+ is required.
```
git clone https://github.com/ronaldosvieira/gym-locm.git
cd gym-locm
pip install -e .
```

## Usage

```python
import gymnasium as gym
import gym_locm

env = gym.make('LOCM-XXX-vX')

terminated = False
truncated = False
while not (terminated or truncated):
    action = ...  # Your agent code here
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
```

## Environments

A match of LOCM has two phases: the **deck-building** phase, where the players build their decks, 
and the **battle** phase, where the playing actually occurs. In LOCM 1.2, the deck-building phase was 
called **draft** phase. In LOCM 1.5, it was called **constructed** phase.

In all environments, by default, a reward of *1* is given if the controlled player wins the battle 
phase, and *-1* otherwise. There are no draws in LOCM. 

### Constructed phase env (LOCM 1.5 only)
```python
env = gym.make("LOCM-constructed-v0")
```

The constructed phase is played. 
Players choose from a card from a card pool of 120 procedurally generated
cards for a total of 30 turns. Players can pick the same card twice, and the card 
pool is the same for both players. They don't know each other's choices.
A default (configurable) policy is used in the battle phase.

**State**: a 120 x 16 matrix (17 features from each of the 120 cards).

**Actions**: 0-119 (representing the index of the card to be chosen).

### Draft phase env (LOCM 1.2 only)
 ```python
env = gym.make("LOCM-draft-v0")
```

The draft phase is played. 
Players alternately choose a card between three randomly sampled cards (without 
replacement) from LOCM 1.2's card pool for a total of 30 turns. The three cards 
from each turn are the same for both players, and they don't know each 
other's choices.
A default (configurable) policy is used in the battle phase.

**State**: a 3 x 16 matrix (16 features from each of the 3 cards). 

**Actions**: 0-2 (chooses first, second or third card).

### Battle phase env
 ```python
env = gym.make("LOCM-battle-v0", version="1.5")
```

```python
env = gym.make("LOCM-battle-v0", version="1.2")
```

The battle phase is played. A default (configurable) policy is used in the draft 
phase. The parameter `version` (default `1.5`) can be used to determine which set
of rules will be used.

**State**: a vector with 3 x 20 + 8 values (16 features from each of the 20 possible cards plus 4 features from each player).

**Actions**: 0-144 (pass, summon, use and attack with all possible origins and targets).
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
     18: USE (card at index 0 of player's hand) (1st creature at player's lane 0)
     19: USE (card at index 0 of player's hand) (2nd creature at player's lane 0)
     20: USE (card at index 0 of player's hand) (3rd creature at player's lane 0)
     21: USE (card at index 0 of player's hand) (1st creature at player's lane 1)
     22: USE (card at index 0 of player's hand) (2nd creature at player's lane 1)
     23: USE (card at index 0 of player's hand) (3rd creature at player's lane 1)
     24: USE (card at index 0 of player's hand) (1st creature at opponent's lane 0)
     25: USE (card at index 0 of player's hand) (2nd creature at opponent's lane 0)
     26: USE (card at index 0 of player's hand) (3rd creature at opponent's lane 0)
     27: USE (card at index 0 of player's hand) (1st creature at opponent's lane 1)
     28: USE (card at index 0 of player's hand) (2nd creature at opponent's lane 1)
     29: USE (card at index 0 of player's hand) (3rd creature at opponent's lane 1)
     30: USE (card at index 1 of player's hand) -1
     31: USE (card at index 1 of player's hand) (1st creature at player's lane 0)
                          ⋮
    120: USE (card at index 7 of player's hand) (3rd creature at opponent's lane 1)
    121: ATTACK (1st creature at player's lane 0) -1
    122: ATTACK (1st creature at player's lane 0) (1st creature at opponent's lane 0)
    123: ATTACK (1st creature at player's lane 0) (2nd creature at opponent's lane 0)
    124: ATTACK (1st creature at player's lane 0) (3rd creature at opponent's lane 0)
    125: ATTACK (2nd creature at player's lane 0) -1
    126: ATTACK (2nd creature at player's lane 0) (1st creature at opponent's lane 0)
                          ⋮
    133: ATTACK (1st creature at player's lane 1) -1
                          ⋮
    144: ATTACK (3rd creature at player's lane 1) (3rd creature at opponent's lane 0)
    
</details>

### Two-player variations
 ```python
env = gym.make("LOCM-draft-2p-v0")
env = gym.make("LOCM-constructed-2p-v0")
env = gym.make("LOCM-battle-2p-v0")
```

Both players are controlled alternately. A reward of *1* is given if the first player wins, and *-1* otherwise. 

### Additional options

Some additional options can be configured at env creation time. These are:

#### Set random seed

This option determines the random seed to be used by the environment. In a match, 
the random seed will be used to generate card choices for each draft turn and to 
shuffle both players' deck at the beginning of the battle. To increase reproducibility,
every time the env is reset, its random state is reset, and `seed + 1` is used as seed.

Usage: `env = gym.make('LOCM-XXX-vX', seed=42)`, default: `None`.

#### Set agents for the roles you don't control

By default, random draft and battle agents are used in the roles not controlled by the 
user (e.g. if it's a single-player draft env, a random agent drafts for the opponent 
player, and random agents battles for both players). To specify different agents for 
these roles, use, for instance:

```python
env = gym.make('LOCM-draft-XXX-vX', draft_agent=RandomDraftAgent(),
                battle_agents=(RandomBattleAgents(), RandomBattleAgents()))
```
```python
env = gym.make('LOCM-constructed-XXX-vX', draft_agent=RandomConstructedAgent(),
                battle_agents=(RandomBattleAgents(), RandomBattleAgents()))
```
```python
env = gym.make('LOCM-battle-XXX-vX', version="1.5", battle_agent=RandomBattleAgent(),
                deck_building_agents=(RandomConstructedAgents(), RandomConstructedAgents()))
```
```python
env = gym.make('LOCM-battle-XXX-vX', version="1.2", battle_agent=RandomBattleAgent(),
                deck_building_agents=(RandomDraftAgents(), RandomDraftAgents()))
```

Trying to specify agents for roles you control will result in an error.

<details>
  <summary>Click to see all available agents</summary>
    
Draft agents:
    
- `PassDraftAgent`: always passes the turn (this is equivalent to always choosing the 
first card).
- `RandomDraftAgent`: drafts at random. 
- `RuleBasedDraftAgent`: drafts like Baseline1 from the Strategy Card Game AI competition.
- `MaxAttackDraftAgent`: drafts like Baseline2 from the Strategy Card Game AI competition.
- `IceboxDraftAgent`: drafts using the card ranking created by CodinGame's user Icebox.
- `ClosetAIDraftAgent`: drafts using the card ranking created by CodinGame's user ClosetAI.
- `UJI1DraftAgent`: drafts like UJIAgent1 from the Strategy Card Game AI competition.
- `UJI2DraftAgent`: drafts like UJIAgent2 from the Strategy Card Game AI competition.
- `CoacDraftAgent`: drafts like Coac from the Strategy Card Game AI competitions pre-2020.
- `NativeDraftAgent`: drafts like an AI player developed for the original LOCM engine, 
whose execution command is passed in the constructor (e.g. `NativeDraftAgent('python3 player.py')`).

Constructed agents:
- `PassConstructedAgent`: always passes the turn (this is equivalent to always choosing the first valid card).
- `RandomConstructedAgent`: chooses any valid card at random.
- `InspiraiConstructedAgent`: constructs like Inspirai from the Strategy Card Game AI competition.

Battle agents:
- `PassBattleAgent`: always passes the turn. 
- `RandomBattleAgent`: chooses any valid action at random (including passing the turn).
- `RuleBasedBattleAgent`: battles like Baseline1 from the Strategy Card Game AI competition.
- `MaxAttackBattleAgent`: battles like Baseline2 from the Strategy Card Game AI competition.
- `GreedyBattleAgent`: battles like Greedy from Kowalski and Miernik's paper <a href="#kowalski2020">[1]</a>.
- `NativeDraftAgent`: battles like an AI player developed for the original LOCM engine, 
whose execution command is passed in the constructor (e.g. `NativeBattleAgent('python3 player.py')`).

If NativeDraftAgent/NativeConstructed and NativeBattleAgent are going to be used to represent the same player,
consider using a single NativeAgent object instead, and passing it as draft and battle agent.
</details>

#### Use item cards

This option determines whether consider green, red and blue item cards in the game. If set to 
false, item cards will not be available to draft, and USE actions will not be 
present on battle envs' action space (ATTACK action codes will start at 17).

Usage: `env = gym.make('LOCM-XXX-vX', items=False)`, default: `True`.

#### Include previously drafted cards to state (draft envs only)

This option determines whether the state of draft envs includes the previously drafted
cards alongside the current card alternatives. If set to true, the state shape of a 
draft env changes from a 3 x 16 matrix to a 33 x 16 matrix, where the 30 first rows 
hold the up to 30 cards drafted in the past turns. The card slots of current and future
picks are filled with zeros.

Usage: `env = gym.make('LOCM-draft-XXX-vX', use_draft_history=True)`, default: `False`.

#### Sort cards in state by mana cost (draft envs only)

This option determines whether the cards in the draft's state matrix will be sorted by 
mana cost in ascending order. This virtually reduces the state space, as every 
possible permutation of three specific cards will result in a single state matrix.

Usage: `env = gym.make('LOCM-draft-XXX-vX', sort_cards=True)`, default: `False`.

#### Change deck length

This option determines the amount of draft/constructed turns that will happen, and, therefore, 
the size of the decks built in the deck building phase. If `use_draft_history` is `True`, the 
state representation in the draft phase will change to accommodate the longer or shorter history 
of past picks.

Usage: `env = gym.make('LOCM-XXX-vX', n=20)`, default: `30`

#### Change amount of cards alternatives per deck building turn

This option determines the amount of cards that will be presented to the players on 
every draft/constructed turn. The state representation and the set of actions in the 
draft/construct phase will change to accommodate the amount of cards options per turn.

Usage: `env = gym.make('LOCM-XXX-vX', k=5)`, default: `120` for LOCM 1.5 and `3` for LOCM 1.2

## Other resources

### Runner

We provide a command-line interface (CLI) to run LOCM matches. It is available as soon as the
repository is installed. Some basic use cases are listed below.

1. Run 1000 matches of LOCM 1.2 in parallel with 4 processes of the Icebox draft agent versus the Coac
draft agent, using random actions in the battle:
    ```bash
    locm-runner --p1-deck-building icebox --p1-battle random \
                --p2-deck-building coac --p2-battle random \
                --games 1000 --version=1.2 --processes 4
    ```

2. Run 1000 matches of LOCM 1.5 of a fully random player against a player developed for the original 
engine, and with a specific random seed:
    ```bash
    locm-runner --p1-deck-building random --p1-battle random \
                --p2-path "python /path/to/agent.py" \
                --games 1000 --version=1.5 --seed 42
    ```
   
Use `locm-runner -h` to see all the available parameters.

### Train draft agents with deep reinforcement learning

We provide scripts to train deep reinforcement learning draft agents as described in 
<a href="#vieira2020a">[2]</a> and <a href="#vieira2020b">[3]</a>. 
Further instructions are available in the README.md file in 
the [experiments](gym_locm/experiments) 
package.

To install the dependencies necessary to run the scripts, install 
the repository with 
```python
pip install -e .['legacy-experiments']
```

We also provide a collection of draft agents trained with deep 
reinforcement learning, and a script to use them in the LOCM's original engine.
Further details on these agents and instructions for the script are available in the
README.md in the 
[trained_models](gym_locm/trained_models) 
package. The use of these draft agents with the Runner script is not implemented yet.

### Train battle agents with deep reinforcement learning

We provide scripts to train deep reinforcement learning battle agents as described in 
<a href="#vieira2022a">[4]</a> and in <a href="vieira2023">[5]</a>. Further instructions are available
in the README.md file in the [experiments/papers/sbgames-2022](gym_locm/experiments/papers/sbgames-2022)
and [experiments/papers/entcom-2023](gym_locm/experiments/papers/entcom-2023) packages.

To install the dependencies necessary to run the scripts, install
the repository with
```python
pip install -e .['experiments']
```

## References
1. <span id="kowalski2023">Kowalski, J., Miernik, R. (2023).
Summarizing Strategy Card Game AI Competition.
IEEE Conference on Games (COG).</span>

2. <span id="kowalski2022">Miernik, R., Kowalski, J. (2022).
Evolving Evaluation Functions for Collectible Card Game AI. 
International Conference on Agents and Artificial Intelligence (ICAART).</span>

3. <span id="kowalski2020">Kowalski, J., Miernik, R. (2020). Evolutionary 
Approach to Collectible Card Game Arena Deckbuilding using Active Genes. 
IEEE Congress on Evolutionary Computation (CEC).</span>

4. <span id="Xi2023">Xi, W., Zhang, Y., Xiao, C., Huang, X., Deng, S., Liang, H., Chen, J., Sun, P. (2023).
Mastering Strategy Card Game (Legends of Code and Magic) via End-to-End Policy and Optimistic Smooth Fictitious Play.
arXiv preprint arXiv:2303.04096.</span>

5. <span id="vieira2020a">Vieira, R., Chaimowicz, L., Tavares, A. R. (2020). Drafting in Collectible Card Games via 
Reinforcement Learning. Master's thesis, Department of Computer Science, Federal University 
of Minas Gerais, Belo Horizonte, Brazil.</span>

6. <span id="vieira2020b">Vieira, R., Tavares, A. R., Chaimowicz, L. (2020). Drafting in 
Collectible Card Games via Reinforcement Learning. 19th Brazilian Symposium of Computer Games
and Digital Entertainment (SBGames).</span>

7. <span id="vieira2022a">Vieira, R. e S., Tavares, A. R., Chaimowicz, L. (2022). Exploring Deep 
Reinforcement Learning for Battling in Collectible Card Games. 19th Brazilian Symposium 
of Computer Games and Digital Entertainment (SBGames).</span>

8. <span id="vieira2023">Vieira, R. e S., Tavares, A. R., Chaimowicz, L. (2023). Towards Sample
Efficient Deep Reinforcement Learning in Collectible Card Games. Entertainment Computing.</span>

## License
[MIT](https://choosealicense.com/licenses/mit/)
