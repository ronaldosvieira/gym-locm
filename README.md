**Status**: Work-in-progress

# gym-locm

[Legends of Code and Magic](https://jakubkowalski.tech/Projects/LOCM/) environment for [OpenAI Gym](https://github.com/openai/gym).


## Environments

1. LOCM-draft-v0 (draft phase only)
2. LOCM-draft-2p-v0 (draft phase only, both players)
3. ~LOCM-battle-v0 (battle phase only)~
4. LOCM-battle-2p-v0 (battle phase only, both players)
5. ~LOCM-v0 (full game)~
6. ~LOCM-2p-v0 (full game, both players)~

(Crossed = not implemented yet)

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
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
