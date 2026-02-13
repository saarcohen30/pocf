# pocf_semi_bandit_experiment.py
"""
Empirical analysis of Algorithm 1 (Surrogate Minimization) under semi-bandit feedback
for the synthetic overlapping-coalition game described by the user.

Key features:
- k = 5 candidate coalitions
- Each agent's action set A_i = [{1,2}, {1,3,5}, {4,5}] (three actions)
- Utilities constructed with Gaussian terms mu_1..mu_5 per sample
- Semi-bandit estimators and exploration bonuses as in Equations (8)-(9)
- Two exploration policies: random and coalitionSize (agent-1 random, others deterministic)
- For outer optimization we use a coordinate-descent / smoothed fictitious-play style solver:
  each agent moves a portion of its mass toward its optimistic pure best response.
- Expectations under a mixed strategy are approximated via Monte Carlo sampling.

Outputs:
- Four PNG figures:
  * duality_gap_vs_n_random.png
  * duality_gap_vs_n_coalitionSize.png
  * duality_gap_vs_M_random.png
  * duality_gap_vs_M_coalitionSize.png

Tuneable hyperparameters (top of file).
"""

import os
import copy
import math
import time
import random
import pickle
from functools import partial
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# -----------------------
# Tunable experiment parameters
# -----------------------
# You can modify these before running experiments.
MC_SAMPLES = 200          # Monte-Carlo samples for expectation estimates (increase for more accuracy)
BR_SAMPLES = 150          # Monte-Carlo samples when computing optimistic best-responses
MAX_ITERS = 80           # maximum number of optimization sweeps over all agents
ALPHA = 0.25              # learning rate toward best-response (0 < ALPHA <= 1). Smaller -> slower updates.
CONV_TOL = 1e-3           # stop early if best objective improves less than this
REPEATS = 5              # number of random seeds per (n, M)
DELTA = 1e-2              # confidence level from the paper
SEED_BASE = 1000          # base seed offset

# Action set (same for all agents)
# Represent each action as a tuple of coalition indices (1-indexed for readability)
ACTION_SET = ( (1,2), (1,3,5), (4,5) )
K = 5  # number of candidate coalitions

# Output directory for results and figures
OUT_DIR = "/content/drive/MyDrive/pocf_results"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Utilities: action/joint sampling
# -----------------------
def sample_agent_action(policy_probs: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an action index (0..2) for one agent given a distribution over actions."""
    return rng.choice(len(ACTION_SET), p=policy_probs)

def sample_joint_action(mixed_strategy: List[np.ndarray], rng: np.random.Generator) -> Tuple[int]:
    """Sample a joint pure action (one action index per agent) from product distribution."""
    return tuple(int(rng.choice(len(ACTION_SET), p=pi)) for pi in mixed_strategy)

# -----------------------
# Utility generation according to the user's construction
# -----------------------
def build_sample_utilities_for_joint_action(joint_action: Tuple[int], rng: np.random.Generator) -> Dict[Tuple[int,int,int], float]:
    """
    For a single joint pure action (tuple of action indices for n agents),
    compute v_{i,j}^ell for all relevant triples (i,j,ell) according to:
      - draw mu_1..mu_5 ~ N(0,1) (one draw per coalition per sample)
      - compute n^ell = number of agents joining coalition ell in this joint action
      - define v_{i,j}^ell as in user's description, clipped to [-1,1].
    Returns a dictionary keyed by (i, j, ell) with float value.
    """
    n = len(joint_action)
    # coalition membership sets
    C = {ell: [] for ell in range(1, K+1)}
    for i, a_idx in enumerate(joint_action):
        action_coalitions = ACTION_SET[a_idx]
        for ell in action_coalitions:
            C[ell].append(i)
    # draw mu_ell for ell=1..5
    mu = {ell: float(rng.normal(0.0, 1.0)) for ell in range(1, K+1)}
    values = {}
    # compute v_{i,j}^ell for pairs that are in same coalition
    for ell in range(1, K+1):
        size = len(C[ell])
        for i in C[ell]:
            for j in C[ell]:
                if i == j:
                    continue
                # formula selection
                if ell == 1:
                    v = - (size) / (n + 1) + mu[1]
                elif ell == 2:
                    v = -1.0 + mu[2]
                elif ell == 3:
                    v = mu[3]
                elif ell == 4:
                    v = -1.0 + mu[4]
                elif ell == 5:
                    v = (size) / (n + 1) + mu[5]
                else:
                    v = 0.0
                # Clip to [-1,1] as in model assumption
                v = max(-1.0, min(1.0, v))
                values[(i, j, ell)] = v
    return values, C

# -----------------------
# Dataset generation
# -----------------------
def generate_offline_dataset(n: int, M: int, policy: str, rng_seed: int) -> Tuple[List[Tuple[int]], Dict[Tuple[int,int,int], List[float]]]:
    """
    Generate a dataset S = { (a^m, v^m) }_m of size M under exploration policy 'policy'.
    policy in {'random', 'coalitionSize'}:
      - 'random': each agent uniformly picks any of the three actions
      - 'coalitionSize': agent 0 (index 0) uniform; agents 1..n-1 deterministically pick action index 1 ({1,3,5})
    Returns:
      - list_joint_actions: list of joint actions (each is tuple of action indices)
      - dict_values: dictionary mapping (i,j,ell) -> list of observed v_{i,j}^{ell,m} across samples
    """
    rng = np.random.default_rng(rng_seed)
    list_actions = []
    dict_values = {}  # accumulate list of v for each triple (i,j,ell)

    for m in range(M):
        # sample joint action
        if policy == 'random':
            a = tuple(int(rng.integers(0, len(ACTION_SET))) for _ in range(n))
        elif policy == 'coalitionSize':
            a = [None] * n
            # agent 0 uniform
            a[0] = int(rng.integers(0, len(ACTION_SET)))
            # others choose {1,3,5} which is ACTION_SET index 1
            for i in range(1, n):
                a[i] = 1
            a = tuple(a)
        else:
            raise ValueError("Unknown policy")
        # compute values for this joint action
        values, _ = build_sample_utilities_for_joint_action(a, rng)
        # store
        list_actions.append(a)
        for key, v in values.items():
            dict_values.setdefault(key, []).append(v)
    return list_actions, dict_values

# -----------------------
# Semi-bandit estimator and bonus from dataset
# -----------------------
def compute_empirical_pair_estimates(n: int, list_actions: List[Tuple[int]], dict_values: Dict[Tuple[int,int,int], List[float]]):
    """
    From dataset, build:
      - N_{i,j}^ell counts
      - hat_v_i^ell(j) empirical means
    Return two dicts keyed by (i,j,ell):
      - counts[(i,j,ell)] -> int
      - means[(i,j,ell)]  -> float (0 if count == 0)
    """
    counts = {}
    means = {}
    # For each triple that appears in dict_values, we have list of values
    for key, vals in dict_values.items():
        counts[key] = len(vals)
        means[key] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    # For triples that didn't appear, counts default to 0 and mean 0
    # There may be (i,j,ell) not present in dict_values if that pair never co-occurred in coalition ell.
    # We leave them absent and treat missing as count=0, mean=0 during queries.
    return counts, means

def hat_v_i_of_joint_action(i: int, joint_action: Tuple[int], means: Dict[Tuple[int,int,int], float]):
    """Compute hat_v_i(a) per equation (8) using precomputed pair means."""
    val = 0.0
    a_idx = joint_action[i]
    a_coalitions = ACTION_SET[a_idx]
    # coalition membership lists
    # compute C_ell quickly
    C = {ell: [] for ell in range(1, K+1)}
    for idx, aind in enumerate(joint_action):
        for ell in ACTION_SET[aind]:
            C[ell].append(idx)
    for ell in a_coalitions:
        for j in C[ell]:
            if j == i:
                continue
            val += means.get((i, j, ell), 0.0)
    return val

def b_i_of_joint_action(i: int, joint_action: Tuple[int], counts: Dict[Tuple[int,int,int], int], delta: float, n: int):
    """Compute exploration bonus b_i^delta(a) per Equation (9)."""
    logterm = math.log(4.0 * (n + 1) * K / delta)
    val = 0.0
    a_idx = joint_action[i]
    a_coalitions = ACTION_SET[a_idx]
    # coalition membership lists
    C = {ell: [] for ell in range(1, K+1)}
    for idx, aind in enumerate(joint_action):
        for ell in ACTION_SET[aind]:
            C[ell].append(idx)
    for ell in a_coalitions:
        for j in C[ell]:
            if j == i:
                continue
            Nij = counts.get((i, j, ell), 0)
            denom = max(1, Nij)
            val += math.sqrt(2.0 * logterm / denom)
    return val

# -----------------------
# Estimating V_i^delta and optimistic best response via Monte Carlo
# -----------------------
def estimate_V_bounds_for_all_agents(mixed_strategy: List[np.ndarray],
                                     counts: Dict[Tuple[int,int,int], int],
                                     means: Dict[Tuple[int,int,int], float],
                                     delta: float,
                                     mc_samples: int,
                                     rng_seed: int):
    """
    Estimate for each agent i:
      - underline_V_i^delta (expected LCB = E[hat_v - b])
      - overline_V_i^delta  (expected UCB = E[hat_v + b])
    by Monte Carlo sampling mc_samples joint actions from mixed_strategy.
    Returns two numpy arrays of length n: (V_lower, V_upper).
    """
    n = len(mixed_strategy)
    rng = np.random.default_rng(rng_seed)
    sum_lower = np.zeros(n, dtype=float)
    sum_upper = np.zeros(n, dtype=float)

    for _ in range(mc_samples):
        a = sample_joint_action(mixed_strategy, rng)
        # For each agent compute hat_v and b
        for i in range(n):
            hat = hat_v_i_of_joint_action(i, a, means)
            b = b_i_of_joint_action(i, a, counts, delta, n)
            sum_lower[i] += (hat - b)
            sum_upper[i] += (hat + b)
    # averages
    V_lower = sum_lower / float(mc_samples)
    V_upper = sum_upper / float(mc_samples)
    return V_lower, V_upper

def compute_optimistic_best_response_for_agent(i: int, mixed_strategy: List[np.ndarray],
                                               counts: Dict[Tuple[int,int,int], int],
                                               means: Dict[Tuple[int,int,int], float],
                                               delta: float,
                                               br_samples: int,
                                               rng_seed: int) -> Tuple[int, float]:
    """
    For agent i, compute optimistic best response (pure action) by maximizing expected UCB:
      argmax_{pure a_i} E_{a_{-i} ~ phi_{-i}} [ overline_v_i^delta( (a_{-i}, a_i) ) ]
    We approximate expectation with Monte Carlo (br_samples).
    Returns (best_action_index, estimated_expected_UCB).
    """
    n = len(mixed_strategy)
    rng = np.random.default_rng(rng_seed)
    # We'll sample others' actions and evaluate candidate a_i choices
    # Pre-generate others' action samples
    others_samples = []
    for _ in range(br_samples):
        # sample a_{-i}
        a_minus = []
        for j in range(n):
            if j == i:
                a_minus.append(None)
            else:
                a_minus.append(int(rng.choice(len(ACTION_SET), p=mixed_strategy[j])))
        others_samples.append(a_minus)
    best_action = None
    best_score = -1e9
    # Try each pure action candidate
    for aidx in range(len(ACTION_SET)):
        total = 0.0
        for samp in others_samples:
            # build full joint action
            a_full = list(samp)
            a_full[i] = aidx
            a_full = tuple(a_full)
            hat = hat_v_i_of_joint_action(i, a_full, means)
            b = b_i_of_joint_action(i, a_full, counts, delta, n)
            total += (hat + b)
        est = total / float(br_samples)
        if est > best_score:
            best_score = est
            best_action = aidx
    return best_action, best_score

# -----------------------
# Surrogate minimization approximate solver
# -----------------------
def surrogate_minimization_approx(n: int,
                                  counts: Dict[Tuple[int,int,int], int],
                                  means: Dict[Tuple[int,int,int], float],
                                  delta: float = DELTA,
                                  mc_samples: int = MC_SAMPLES,
                                  br_samples: int = BR_SAMPLES,
                                  max_iters: int = MAX_ITERS,
                                  alpha: float = ALPHA,
                                  rng_seed: int = 0):
    """
    Approximate solution to min_phi max_i [ overlineV_i^*(phi_-i) - underlineV_i(phi) ]
    using smoothed coordinate descent / fictitious-play-like iterations.
    Returns:
      - phi_out: list of per-agent arrays (length 3) - final mixed strategy
      - best_gap_est: estimated surrogate gap value for phi_out (Monte Carlo)
      - diagnostics: dictionary with iterative history (optional)
    """
    rng = np.random.default_rng(rng_seed)
    # initialize phi as uniform for each agent
    phi = [np.ones(len(ACTION_SET)) / len(ACTION_SET) for _ in range(n)]
    best_phi = copy.deepcopy(phi)
    # compute initial bounds
    V_lower, V_upper = estimate_V_bounds_for_all_agents(phi, counts, means, delta, mc_samples, rng_seed + 1)
    # compute optimistic best responses values for each agent
    # For each agent i compute best_response_i and its optimistic expected value wrt phi_-i
    br_values = np.zeros(n, dtype=float)
    for i in range(n):
        _, br_val = compute_optimistic_best_response_for_agent(i, phi, counts, means, delta, br_samples, rng_seed + 10 + i)
        br_values[i] = br_val
    hat_gap = np.max(br_values - V_lower)
    best_gap = hat_gap
    diagnostics = {'gaps': [hat_gap]}

    for it in range(max_iters):
        improved = False
        # sweep over agents
        for i in range(n):
            best_action, best_val = compute_optimistic_best_response_for_agent(i, phi, counts, means, delta, br_samples, rng_seed + 1000 + it * n + i)
            # update phi_i toward one-hot(best_action)
            one_hot = np.zeros(len(ACTION_SET), dtype=float)
            one_hot[best_action] = 1.0
            new_phi_i = (1.0 - alpha) * phi[i] + alpha * one_hot
            phi[i] = new_phi_i
        # estimate new bounds and best-response values
        V_lower, V_upper = estimate_V_bounds_for_all_agents(phi, counts, means, delta, mc_samples, rng_seed + 2000 + it)
        for i in range(n):
            _, br_val = compute_optimistic_best_response_for_agent(i, phi, counts, means, delta, br_samples, rng_seed + 3000 + it * n + i)
            br_values[i] = br_val
        hat_gap = float(np.max(br_values - V_lower))
        diagnostics['gaps'].append(hat_gap)
        # keep best
        if hat_gap + 1e-12 < best_gap:
            best_gap = hat_gap
            best_phi = copy.deepcopy(phi)
            improved = True
        # stopping condition
        if it > 0:
            if abs(diagnostics['gaps'][-2] - diagnostics['gaps'][-1]) < CONV_TOL:
                # small improvement, break
                break
        # optional small safeguard
    # final estimate for best_phi
    V_lower, V_upper = estimate_V_bounds_for_all_agents(best_phi, counts, means, delta, mc_samples, rng_seed + 7777)
    # recompute optimistic best responses one last time
    final_br_values = np.zeros(n, dtype=float)
    for i in range(n):
        _, br_val = compute_optimistic_best_response_for_agent(i, best_phi, counts, means, delta, br_samples, rng_seed + 9000 + i)
        final_br_values[i] = br_val
    final_gap = float(np.max(final_br_values - V_lower))
    return best_phi, final_gap, diagnostics

# -----------------------
# Single experiment run (one dataset and solver execution)
# -----------------------
def run_single_experiment(n: int, M: int, policy: str, seed: int):
    """
    Runs a single repetition:
      - generate dataset (list_actions, dict_values)
      - compute counts & means
      - run surrogate minimization approx solver
      - return the final estimated duality gap
    """
    # generate dataset
    print(f"Running experiment n={n}, M={M}, policy={policy}, seed={seed} ...")
    print("Generating dataset...")
    list_actions, dict_values = generate_offline_dataset(n, M, policy, seed)
    print("Computing empirical estimates...")
    counts, means = compute_empirical_pair_estimates(n, list_actions, dict_values)
    # run solver
    print("Running surrogate minimization approximate solver...")
    phi_out, gap_est, diag = surrogate_minimization_approx(n, counts, means, delta=DELTA,
                                                           mc_samples=MC_SAMPLES,
                                                           br_samples=BR_SAMPLES,
                                                           max_iters=MAX_ITERS,
                                                           alpha=ALPHA,
                                                           rng_seed=seed + 12345)
    a = sample_joint_action(phi_out, np.random.default_rng(seed + 12345))
    print(a)
    return gap_est

# -----------------------
# Experiment orchestration
# -----------------------
def run_grid_experiments(n_grid, M_grid, policy, repeats=REPEATS, parallel_jobs=1):
    """
    For the given exploration policy, run:
      - varying n over n_grid with fixed M = M_grid_fixed (if M_grid is scalar)
      - or varying M over M_grid with fixed n = n_grid_fixed (if n_grid is scalar)
    The caller will provide appropriate grids.
    Returns a dictionary of results.
    """
    results = {}
    # Two cases: one of n_grid or M_grid might be a single fixed value
    # We accept both as lists. Caller decides how to pass grids.
    # We'll run each tuple (n, M) repeating 'repeats' seeds.
    tasks = []
    for n in n_grid:
        for M in M_grid:
            tasks.append((n, M))
    # run sequentially or in parallel
    def run_task(nMtuple):
        n, M = nMtuple
        seed_base_local = SEED_BASE + (n * 1000 + M) % 999999
        gaps = []
        for r in range(repeats):
            seed = seed_base_local + r
            gap = run_single_experiment(n, M, policy, seed)
            gaps.append(gap)
        gaps = np.array(gaps)

        all_results = {}
        all_results[(n, M)] = {
            'gaps': gaps,
            'mean': float(np.mean(gaps)),
            'std': float(np.std(gaps, ddof=1))
        }
        savepath = os.path.join(OUT_DIR, f"results_{policy}_n_{n}_M_{M}.pkl")
        with open(savepath, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Saved results to {savepath}")

        return (n, M, gaps)
    if parallel_jobs == 1:
        out = [run_task(t) for t in tasks]
    else:
        print("Running experiments in parallel with", parallel_jobs, "jobs...")
        out = Parallel(n_jobs=parallel_jobs)(delayed(run_task)(t) for t in tasks)
    # gather
    for n, M, gaps in out:
        results[(n, M)] = {
            'gaps': gaps,
            'mean': float(np.mean(gaps)),
            'std': float(np.std(gaps, ddof=1))
        }
    return results

# -----------------------
# Plotting helpers
# -----------------------
def plot_varying_n(results: Dict, n_values: List[int], M_fixed: int, policy_name: str, filename: str):
    means = []
    stds = []
    for n in n_values:
        r = results[(n, M_fixed)]
        means.append(r['mean'])
        stds.append(r['std'])
    means = np.array(means)
    stds = np.array(stds)
    plt.figure(figsize=(8,5))
    plt.plot(n_values, means, marker='o', linestyle='-', label='mean duality gap')
    plt.fill_between(n_values, means - stds, means + stds, color='lightblue', alpha=0.6)
    plt.xlabel('Number of agents n')
    plt.ylabel('Duality gap (surrogate estimate)')
    plt.title(f'Duality gap vs n  — policy: {policy_name}  (M={M_fixed})')
    plt.grid(True)
    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, filename)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved {outpath}")

def plot_varying_M(results: Dict, M_values: List[int], n_fixed: int, policy_name: str, filename: str):
    means = []
    stds = []
    for M in M_values:
        r = results[(n_fixed, M)]
        means.append(r['mean'])
        stds.append(r['std'])
    means = np.array(means)
    stds = np.array(stds)
    plt.figure(figsize=(8,5))
    plt.plot(M_values, means, marker='o', linestyle='-', label='mean duality gap')
    plt.fill_between(M_values, means - stds, means + stds, color='lightblue', alpha=0.6)
    plt.xlabel('Dataset size M')
    plt.ylabel('Duality gap (surrogate estimate)')
    plt.title(f'Duality gap vs M  — policy: {policy_name}  (n={n_fixed})')
    plt.grid(True)
    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, filename)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved {outpath}")

# -----------------------
# Main driver that matches user's requested experiments
# -----------------------
def main(parallel_jobs=1, run_all=True):
    """
    Orchestrates the experiments and plotting exactly as requested:
    - For each exploration policy, produce:
        1) Varying n in {5,25,50,75,100} with fixed M=100 (5 seeds each)
        2) Varying M in {10,20,30,50,60,70,80,90,100} with fixed n=10 (5 seeds each)
    Adjust MC_SAMPLES, BR_SAMPLES, MAX_ITERS, ALPHA at top of file to trade-off speed/accuracy.
    """
    # grids as requested
    n_grid = [5,10,25]
    # M_fixed = 100
    # M_grid1 = [M_fixed]
    M_grid2_list = [100, 10000, 30000, 50000, 70000, 100000]
    # n_fixed = 10

    policies = [('random', 'Random Exploration'), ('coalitionSize', 'CoalitionSize')]
    for policy, policy_name in policies:
        # # 1) vary n, fixed M=100
        # tasks_n = [(n, M_fixed) for n in n_grid]
        # print(f"Running varying n for policy {policy_name} ...")
        # results_n = run_grid_experiments(n_grid=n_grid, M_grid=M_grid1, policy=policy, repeats=REPEATS, parallel_jobs=parallel_jobs)
        # # results_n keys are (n, M_fixed)
        # plot_varying_n(results_n, n_grid, M_fixed, policy_name, f"duality_gap_vs_n_{policy}.png")

        # 2) vary M, fixed n
        for n_fixed in n_grid:
          print(f"Running varying M for policy {policy_name} and n={n_fixed} agents ...")
          results_M = run_grid_experiments(n_grid=[n_fixed], M_grid=M_grid2_list, policy=policy, repeats=REPEATS, parallel_jobs=parallel_jobs)
          plot_varying_M(results_M, M_grid2_list, n_fixed, policy_name, f"duality_gap_vs_M_{policy}_and_{n_fixed}_agents.png")

          # save raw results
          # all_results = {**results_n, **results_M}
          all_results = {**results_M}
          savepath = os.path.join(OUT_DIR, f"results_{policy}_n_is_{n_fixed}.pkl")
          with open(savepath, "wb") as f:
              pickle.dump(all_results, f)
          print(f"Saved results to {savepath}")

if __name__ == "__main__":
    # If executed as a script, run the full experiment.
    # Warning: this can take considerable time depending on MC_SAMPLES, BR_SAMPLES, n and M.
    start = time.time()
    main(parallel_jobs=1, run_all=True)
    end = time.time()
    print("Total time (s):", end - start)