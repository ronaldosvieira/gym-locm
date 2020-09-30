# Experiments

This package contains the source-code for all experiments performed for our thesis <a href="#vieira2020a">[1]</a> 
and paper <a href="#vieira2020b">[2]</a>. They consist of: (i) a hyperparameter search script, (ii) a training 
script, (iii) a tournament runner script and a (iv) similarities script. Th steps to reproduce each experiment are
described as follows.


### How to reproduce the paper/thesis experiments

#### Section 5.3 or IV.D: Hyperparameter tuning

to do

#### Section 5.4 or IV.E: Comparison between approaches

to do

#### Section 5.5 or IV.F: Comparison with other draft strategies

to do

#### Section 5.6 or IV.G: Agent improvement in the CoG 2019 LOCM tournament

We used the 
[source code of the Strategy Card Game AI competition](https://github.com/acatai/Strategy-Card-Game-AI-Competition/tree/master/contest-2019-08-COG) 
to re-run the matches, replacing the *max-attack* player (named Baseline2) with a personalized player featuring 
our best draft agent and the battle portion on the *max-attack* player. This can be reproduced by altering line 
11 of the runner script 
([run.sh](https://github.com/acatai/Strategy-Card-Game-AI-Competition/blob/master/contest-2019-08-COG/run.sh))
from `AGENTS[10]="python3 Baseline2/main.py"` to
```bash
AGENTS[10]="python3 gym_locm/toolbox/predictor.py --battle \"python3 Baseline2/main.py\" \
    --draft-1 path/to/gym_locm/trained_models/max-attack/immediate/1st/6.json \
    --draft-2 path/to/gym_locm/trained_models/max-attack/immediate/2nd/8.json
```
then, executing it. Save the output to a text file in the same folder, then run 
[analyze.py](https://github.com/acatai/Strategy-Card-Game-AI-Competition/blob/master/contest-2019-08-COG/analyze.py) 
to extract win rates.
See the [trained_models](https://github.com/ronaldosvieira/gym-locm/tree/master/gym_locm/trained_models) 
package for more information on the predictor script.

### References

1. <span id="vieira2020a">Vieira, R., Chaimowicz, L., Tavares, A. R. (2020.) Drafting in Collectible Card Games via 
Reinforcement Learning. Master's thesis, Department of Computer Science, Federal University 
of Minas Gerais, Belo Horizonte, Brazil.</span>

2. <span id="vieira2020b">Vieira, R., Tavares, A. R., Chaimowicz, L. (2020). Drafting in 
Collectible Card Games via Reinforcement Learning. 19th Brazilian Symposium of Computer Games
and Digital Entertainment (SBGames).</span>
