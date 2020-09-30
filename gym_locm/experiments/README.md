# Experiments

This package contains the source-code for all experiments performed for our thesis <a href="#vieira2020a">[1]</a> 
and paper <a href="#vieira2020b">[2]</a>. They consist of: (i) a hyperparameter search script, (ii) a training 
script, (iii) a tournament runner script and a (iv) similarities script. Th steps to reproduce each experiment are
described as follows.


### How to reproduce the paper/thesis experiments

#### Section 5.3 or IV.D: Hyperparameter tuning

To perform a hyperparameter tuning, simply execute the [hyp-search.py](hyp-search.py) script:

```
python3 gym_locm/experiments/hyp-search.py --approach <approach> --battle-agent <battle_agent> \
    --path hyp_search_results/ --seed 96765 --processes 4
```

The list and range of hyperparameted explored is available in the Appendix of our paper and in Attachment A of 
our thesis. we performed hyperparameter tunings for all combinations of `<approach>` (`immediate`, `history` 
and `lstm`) and `<battle_agent>` (`max-attack` and `greedy`). Each run of the script took around 2 days with the
`max-attack` battle agent and more than a week with the `greedy` battle agent. To learn about other script's 
parameters, execute it with the `--help` flag.

#### Section 5.4 or IV.E: Comparison between approaches

to do

#### Section 5.5 or IV.F: Comparison with other draft strategies

To run the tournament, simply execute the [tournament.py](tournament.py) script:
```
python3 gym_locm/experiments/tournament.py \
    --drafters random max-attack coac closet-ai icebox \
        gym_locm/trained_models/<battle_agent>/immediate/ \
        gym_locm/trained_models/<battle_agent>/history/ \
        gym_locm/trained_models/<battle_agent>/lstm/ \
    --battler <battle_agent> --concurrency 4 --games 1000 --path tournament_results/ \
    --seeds 32359627 91615349 88803987 83140551 50731732 19279988 35717793 48046766 86798618 62644993
```
replacing `<battle_agent>` for either `max-attack` or `greedy`, respectively, to run either tournament as 
depicted in the thesis. The tournament results include matches of all draft agents versus the `max-attack`
draft agent, as depicted in the paper. The script will create files at `tournament_results/` describing 
the individual win rates of every set of matches, the aggregate win rates, average mana curves and every 
individual draft choice made by every agent, in CSV format, for human inspection, and as serialized Pandas 
data frames (PKL format), for easy further data manipulation. To learn about other script's 
parameters, execute it with the `--help` flag.

To reproduce the plot containing the agent's three-dimensional coordinates found via Principal Component 
Analysis and grouped via K-Means, simply execute the [similarities.py](similarities.py) script:
```
python3 gym_locm/experiments/similarities.py \
  --files max_attack_tournament_results/choices.csv greedy_tournament_results/choices.csv
```
which will result in a PNG image saved to the current folder.

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
    --draft-2 path/to/gym_locm/trained_models/max-attack/immediate/2nd/8.json"
```
then, executing it. Paralellism can be achieved by running the script in multiple processes/machines. Save the 
output to text files named `out-*.txt` (with a number instead of `*`) in the same folder, then run 
[analyze.py](https://github.com/acatai/Strategy-Card-Game-AI-Competition/blob/master/contest-2019-08-COG/analyze.py) 
to extract win rates. The runner script can take up to several days, and the analyze script can take up to some hours.
See the [trained_models](https://github.com/ronaldosvieira/gym-locm/tree/master/gym_locm/trained_models) 
package for more information on the predictor script.

### References

1. <span id="vieira2020a">Vieira, R., Chaimowicz, L., Tavares, A. R. (2020). Drafting in Collectible Card Games via 
Reinforcement Learning. Master's thesis, Department of Computer Science, Federal University 
of Minas Gerais, Belo Horizonte, Brazil.</span>

2. <span id="vieira2020b">Vieira, R., Tavares, A. R., Chaimowicz, L. (2020). Drafting in 
Collectible Card Games via Reinforcement Learning. 19th Brazilian Symposium of Computer Games
and Digital Entertainment (SBGames).</span>
