# Offline Learning of Nash Stable Coalition Structures with Possibly Overlapping Coalitions

Code for implementation of surrogate minimization in possibly overlapping coalition formation (POCF) games under bandit and semi-bandit feedback.
Given a fixed dataset of samples composed of a coalition structure and utility feedbacks resulting from that coalition structure, surrogate minimization in POCF games learns an approximate Nash stable coalition strucutre.

If any part of this code is used, the following paper must be cited: 

**Saar Cohen. Offline Learning of Nash Stable Coalition Structures with Possibly Overlapping Coalitions. <em>In AAMAS'26: the 25th International Conference on Autonomous Agents and Multiagent Systems, 2026</em> (To Appear).**

## Experiments

We applied our algorithms to several synthetic classes of randomly generated games.

After a coalition strucutre determined for a certain sample, `pocf_bandit_unif_guass.py` implements the following classes of bandit feedbacks. Specifically, for each pair of agents `i` and `j` who are in the same coalition
1. **Size-Dependent Uniform**: The mutual utility of agents `i` and `j` is first sampled uniformly at random from [-1,1] and then multiplied by the size of their coalition, divided by the total number of agents plus 1.
2. **Size-Independent Uniform**: The mutual utility of agents `i` and `j` is sampled uniformly at random from [-1,1].
3. **Size-Dependent Gaussian**: The mutual utility of agents `i` and `j` is first drawn from a Gaussian distribution and then multiplied by the size of their coalition, divided by the total number of agents plus 1.
4. **Size-Independent Gaussian**: The mutual utility of agents `i` and `j` is drawn from a Gaussian distribution.

`pocf_semi_bandit_unif_guass_experiment.py` implements those experiments under semi-bandit feedback.

In contrast, `pocf_semi_bandit_Braess_experiment.py` implements a mixed size effects, constructing stylized games where coalitions differ in how their sizes affect agents’ utilities. In particular, our designed games reflect common trade-offs in coalition formation: agents must decide whether to join small, high-quality groups to avoid overcrowding, or larger, more cooperative ones that benefit from coordination, while avoiding consistently poor environments. It thereby provides a controlled and interpretable testbed for evaluating our algorithm.

## Execution of Experiments under Bandit Feedback
To execute `pocf_bandit_unif_guass.py`, run the following:
`python pocf_bandit_unif_guass.py --single-model <MODEL_NAME> --policy <POLICY_NAME> --impact <X> --parallel <NUM_OF_JOBS>`

# <MODEL_NAME> 
Allows the user to run only a single utility model. Utility generation models supported (pass model string):
- 'uniform'             : uniform $v_{i,j} in [-1,1]$.
- 'gaussian'            : base \$mu_{i,j} \sim U(-1,1), v \sim N(mu, sd)$ where $sd$ depends on $mu$.
- 'size_uniform'        : $x \sim U(-1,1), v = x * size / (n+1)$
- 'size_gaussian'       : $mu \sim U(-1,1), x \sim N(mu, sd), v = x * size / (n+1)$

# <POLICY_NAME>
Generate $M$ samples (joint actions and semi-bandit reports) under exploration policy:
- 'random': each agent picks uniformly among ACTION_SET
- 'oneRand': agent 0 uniform; others deterministic pick ACTION_SET index 1 if available

# <X>
- `n` - Impact of number of agents $n$
- `k` - Impact of number of coalitions $k$ ($n=10$)

# <NUM_OF_JOBS>
Number of parallel jobs for grid tasks (joblib).

## Execution of Experiments under Semi-Bandit Feedback
To execute `pocf_semi_bandit_unif_guass_experiment.py`, run the following with similar arguments:
`python pocf_semi_bandit_unif_guass_experiment.py --single-model <MODEL_NAME> --impact <X> --parallel <NUM_OF_JOBS>`

## Execution of Mixed Coalition-Size-Effects
`python pocf_semi_bandit_Braess_experiment.py`
