# Trained models

In this folder, there are all draft agents trained and used in experiments for our paper and thesis. They are organized in the following folder structure:
```
.../trained_models/<battle_agent>/<draft_approach>/<1st or 2nd player>/<file>.(zip|json)
```

Where: 
- `battle_agent` means which battle agent played the battles while they were being trained,
- `draft_approach` is either `immediate` (disregards past picks), `history` (leverages past picks by enumerating them in the input) or `lstm` 
(leverages past picks via long short-term memory (LSTM) units). The `immediate` obtained best results.
- `1st or 2nd player` is either `1st` or `2nd`, meaning for which role they were trained.
- `file` is a number from 1 to 10. For each combination of battle agent, draft approach and role, we trained ten different agents.

The models are represented as ZIP files, which can be loaded by the [stable-baselines](https://github.com/hill-a/stable-baselines) library, 
and JSON files, which can be used in our [predictor](https://github.com/ronaldosvieira/gym-locm/blob/master/gym_locm/toolbox/predictor.py) script.

### How to use our draft agents

The only use case implemented so far is replacing the draft portion of an AI player developed for 
[LOCM's original engine](https://github.com/acatai/Strategy-Card-Game-AI-Competition/tree/master/referee-java) with one of our trained draft agents.
To do so, execute `predictor.py` in one of the two scenarios below:
1. To use **different** models when playing first and second, with some AI player `player.py`.
  ```
  python3 gym_locm/toolbox/predictor.py --draft-1 gym_locm/trained_models/greedy/immediate/1st/4.json \
                                        --draft-2 gym_locm/trained_models/greedy/immediate/2nd/3.json \
                                          --battle "python3 /path/to/player.py"
  ```
2. To use the **same** model when playing first and second, with some AI player `player`
  ```
  python3 gym_locm/toolbox/predictor.py --draft gym_locm/trained_models/max-attack/history/1st/5.json \
                                        --battle "./path/to/player"
  ```
The use of LSTM draft agents with predictor is not yet implemented.
