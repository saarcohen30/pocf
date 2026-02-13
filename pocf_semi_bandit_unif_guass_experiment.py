# pocf_semi_bandit_exp_all_models.py
"""
Empirical analysis for Algorithm 1 (Surrogate Minimization) under semi-bandit feedback,
extended with multiple utility-generation models, random ACTION_SET per K, and
two requested experiment plots per utility model:

1) Impact of number of agents n (curves for n in [5,10,25,50,75]) over dataset sizes
   [100, 10000, 30000, 50000, 70000, 100000]. Uses K=5 for this plot.

2) For n=25, impact of number of coalitions K in [5,10,15,20,25], over same dataset sizes.

Utility generation models supported (pass model string):
 - 'uniform'             : uniform v_{i,j} in [-1,1].
 - 'gaussian'            : base mu_{i,j} ~ U(-1,1), v ~ N(mu, sd) where sd depends on mu.
 - 'size_uniform'        : x ~ U(-1,1), v = x * size / (n+1)
 - 'size_gaussian'       : mu ~ U(-1,1), x ~ N(mu, sd), v = x * size / (n+1)

The code uses an approximate solver for the surrogate minimization:
 - optimistic best responses are searched among pure actions (consistent with paper).
 - mixed-strategy optimization is approximated via smoothed best-response updates.
 - expectations are approximated via Monte Carlo sampling.
"""

import os
import math
import time
import pickle
import random
import itertools
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# -----------------------
# User-tunable global experiment parameters
# -----------------------
MC_SAMPLES = 50          # Monte-Carlo samples for expectation estimates
BR_SAMPLES = 50          # Monte-Carlo samples for optimistic best-response estimation
MAX_ITERS = 80           # max sweeps over agents in surrogate minimization
ALPHA = 0.25              # step-size toward pure best response in each agent update
CONV_TOL = 1e-3           # stopping tolerance on surrogate objective improvement
REPEATS = 5              # number of different random seeds per (n,M)
DELTA = 1e-2              # confidence level for bonus (as in paper)
SEED_BASE = 123456        # base for reproducible seeds
ACTION_SET_SIZE = 3       # number of actions (subsets) per agent (randomly generated)
OUT_DIR = "/content/drive/MyDrive/pocf_results_all_models"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Utility-generation model names
# -----------------------
UTILITY_MODELS = ['uniform', 'gaussian', 'size_uniform', 'size_gaussian']

# -----------------------
# Helper: generate random ACTION_SET
# -----------------------
def generate_random_action_set(K: int, action_set_size: int, rng: np.random.Generator) -> Tuple[Tuple[int,...], ...]:
    """
    Generate `action_set_size` distinct non-empty subsets of {1,...,K} uniformly at random.
    Returns a tuple of actions; each action is a tuple of coalition indices (1-indexed).
    """
    assert K >= 1 and action_set_size >= 1
    universe = list(range(1, K + 1))
    all_nonempty_subsets = []
    # Instead of enumerating all (which is 2^K -1), sample uniformly until we get enough unique
    seen = set()
    actions = []
    # If K is small, we can draw from all possibilities to ensure uniformity:
    if K <= 20 and (2**K - 1) <= 200000:
        # build full list
        for r in range(1, K + 1):
            for comb in itertools.combinations(universe, r):
                all_nonempty_subsets.append(comb)
        rng.shuffle(all_nonempty_subsets)
        # take first action_set_size unique ones
        # If action_set_size > available combos, allow duplicates by repeating
        for idx in range(action_set_size):
            actions.append(all_nonempty_subsets[idx % len(all_nonempty_subsets)])
    else:
        # K large: sample subsets by randomly including each element with p=0.5 until we have enough unique
        while len(actions) < action_set_size:
            subset = tuple(sorted([u for u in universe if rng.random() < 0.5]))
            if len(subset) == 0:
                # ensure non-empty
                subset = (rng.integers(1, K+1),)
            if subset not in seen:
                seen.add(subset)
                actions.append(subset)
        # If we somehow didn't get enough (unlikely), pad with singletons
        while len(actions) < action_set_size:
            actions.append((rng.integers(1, K+1),))
    return tuple(actions)

# -----------------------
# Build sample utilities for a joint action under different models
# -----------------------
def build_sample_utilities_for_joint_action(joint_action: Tuple[int],
                                           K: int,
                                           ACTION_SET: Tuple[Tuple[int,...], ...],
                                           rng: np.random.Generator,
                                           utility_model: str):
    """
    For a single joint action (action index per agent referencing ACTION_SET),
    compute the semi-bandit pairwise utilities v_{i,j}^ell for all pairs that appear
    in the same coalition ell under this joint action, according to chosen model.

    Returns:
      - values: dict keyed by (i, j, ell) -> float in [-1,1]
      - C: dict ell -> list of agent indices in coalition ell
    """
    n = len(joint_action)
    # coalition membership lists
    C = {ell: [] for ell in range(1, K+1)}
    for i, a_idx in enumerate(joint_action):
        action_coalitions = ACTION_SET[a_idx]
        for ell in action_coalitions:
            if 1 <= ell <= K:
                C[ell].append(i)

    values = {}

    if utility_model == 'uniform':
        # For each coalition, for each ordered pair i != j in that coalition, sample v ~ U(-1,1)
        for ell, members in C.items():
            for i in members:
                for j in members:
                    if i == j:
                        continue
                    v = float(rng.uniform(-1.0, 1.0))
                    v = max(-1.0, min(1.0, v))
                    values[(i, j, ell)] = v

    elif utility_model == 'gaussian':
        # For each ordered pair (i,j) that co-occur, sample base mu_{i,j} ~ U(-1,1), sd depends on mu
        for ell, members in C.items():
            for i in members:
                for j in members:
                    if i == j:
                        continue
                    mu = float(rng.uniform(-1.0, 1.0))
                    sd = (1.0 - mu) if mu >= 0.0 else abs(-1.0 - mu)
                    # ensure sd positive (it should be except pathological mu)
                    sd = max(1e-6, sd)
                    v = float(rng.normal(loc=mu, scale=sd))
                    v = max(-1.0, min(1.0, v))
                    values[(i, j, ell)] = v

    elif utility_model == 'size_uniform':
        # For each coalition, draw x ~ U(-1,1) per pair and scale by size/(n+1)
        for ell, members in C.items():
            size = len(members)
            scale = size / (n + 1.0)
            for i in members:
                for j in members:
                    if i == j:
                        continue
                    x = float(rng.uniform(-1.0, 1.0))
                    v = x * scale
                    v = max(-1.0, min(1.0, v))
                    values[(i, j, ell)] = v

    elif utility_model == 'size_gaussian':
        # For each pair co-occurring, sample mu_{i,j}~U(-1,1), sd as earlier, sample x~N(mu,sd), then v = x * size/(n+1)
        for ell, members in C.items():
            size = len(members)
            scale = size / (n + 1.0)
            for i in members:
                for j in members:
                    if i == j:
                        continue
                    mu = float(rng.uniform(-1.0, 1.0))
                    sd = (1.0 - mu) if mu >= 0.0 else abs(-1.0 - mu)
                    sd = max(1e-6, sd)
                    x = float(rng.normal(loc=mu, scale=sd))
                    v = x * scale
                    v = max(-1.0, min(1.0, v))
                    values[(i, j, ell)] = v

    else:
        raise ValueError(f"Unknown utility_model: {utility_model}")

    return values, C

# -----------------------
# Dataset generation (accepts ACTION_SET, K, utility model)
# -----------------------
def generate_offline_dataset(n: int, M: int, policy: str, rng_seed: int,
                             K: int, ACTION_SET: Tuple[Tuple[int,...], ...],
                             utility_model: str):
    """
    Generate M samples (joint actions and semi-bandit reports) under exploration policy:
      - 'random': each agent picks uniformly among ACTION_SET
      - 'coalitionSize': agent 0 uniform; others deterministic pick ACTION_SET index 1 if available
    Returns:
      - list_actions: list of joint_action tuples
      - dict_values: dict (i,j,ell) -> list of observed v's across samples
    """
    rng = np.random.default_rng(rng_seed)
    list_actions = []
    dict_values = {}

    for m in range(M):
        if policy == 'random':
            a = tuple(int(rng.integers(0, len(ACTION_SET))) for _ in range(n))
        elif policy == 'coalitionSize':
            a = [None] * n
            a[0] = int(rng.integers(0, len(ACTION_SET)))
            # ensure ACTION_SET has index 1; if not, pick index 0 deterministically for others
            default_index = 1 if len(ACTION_SET) > 1 else 0
            for i in range(1, n):
                a[i] = default_index
            a = tuple(a)
        else:
            raise ValueError("Unknown policy")

        values, _ = build_sample_utilities_for_joint_action(a, K, ACTION_SET, rng, utility_model)
        list_actions.append(a)
        for key, v in values.items():
            dict_values.setdefault(key, []).append(v)

    return list_actions, dict_values

# -----------------------
# Semi-bandit estimators & bonuses (counts & means)
# -----------------------
def compute_empirical_pair_estimates(n: int, list_actions: List[Tuple[int]], dict_values: Dict[Tuple[int,int,int], List[float]]):
    """
    Compute counts N_{i,j}^ell and empirical means hat_v_i^ell(j).
    """
    counts = {}
    means = {}
    for key, vals in dict_values.items():
        counts[key] = len(vals)
        means[key] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    return counts, means

def hat_v_i_of_joint_action(i: int, joint_action: Tuple[int], ACTION_SET: Tuple[Tuple[int,...], ...], means: Dict[Tuple[int,int,int], float]):
    """Compute hat{v}_i(a) per Equation (8) using precomputed pair means. ACTION_SET provided."""
    val = 0.0
    a_idx = joint_action[i]
    a_coalitions = ACTION_SET[a_idx]
    # compute coalition membership lists
    C = {}
    for idx, aind in enumerate(joint_action):
        for ell in ACTION_SET[aind]:
            C.setdefault(ell, []).append(idx)
    for ell in a_coalitions:
        for j in C.get(ell, []):
            if j == i:
                continue
            val += means.get((i, j, ell), 0.0)
    return val

def b_i_of_joint_action(i: int, joint_action: Tuple[int], ACTION_SET: Tuple[Tuple[int,...], ...],
                        counts: Dict[Tuple[int,int,int], int], delta: float, n: int):
    """Compute exploration bonus b_i^delta(a) per Equation (9)."""
    logterm = math.log(4.0 * (n + 1) * max(1, max([max(a) for a in ACTION_SET])) / delta)  # safe upper bound of k
    val = 0.0
    a_idx = joint_action[i]
    a_coalitions = ACTION_SET[a_idx]
    # coalition membership lists
    C = {}
    for idx, aind in enumerate(joint_action):
        for ell in ACTION_SET[aind]:
            C.setdefault(ell, []).append(idx)
    for ell in a_coalitions:
        for j in C.get(ell, []):
            if j == i:
                continue
            Nij = counts.get((i, j, ell), 0)
            denom = max(1, Nij)
            val += math.sqrt(2.0 * logterm / denom)
    return val

# -----------------------
# Monte Carlo estimators for V bounds and optimistic BR
# -----------------------
def estimate_V_bounds_for_all_agents(mixed_strategy: List[np.ndarray],
                                     counts: Dict[Tuple[int,int,int], int],
                                     means: Dict[Tuple[int,int,int], float],
                                     ACTION_SET: Tuple[Tuple[int,...], ...],
                                     delta: float,
                                     mc_samples: int,
                                     rng_seed: int):
    n = len(mixed_strategy)
    rng = np.random.default_rng(rng_seed)
    sum_lower = np.zeros(n, dtype=float)
    sum_upper = np.zeros(n, dtype=float)

    for _ in range(mc_samples):
        a = tuple(int(rng.choice(len(ACTION_SET), p=pi)) for pi in mixed_strategy)
        for i in range(n):
            hat = hat_v_i_of_joint_action(i, a, ACTION_SET, means)
            b = b_i_of_joint_action(i, a, ACTION_SET, counts, delta, n)
            sum_lower[i] += (hat - b)
            sum_upper[i] += (hat + b)
    V_lower = sum_lower / float(mc_samples)
    V_upper = sum_upper / float(mc_samples)
    return V_lower, V_upper

def compute_optimistic_best_response_for_agent(i: int, mixed_strategy: List[np.ndarray],
                                               counts: Dict[Tuple[int,int,int], int],
                                               means: Dict[Tuple[int,int,int], float],
                                               ACTION_SET: Tuple[Tuple[int,...], ...],
                                               delta: float,
                                               br_samples: int,
                                               rng_seed: int) -> Tuple[int, float]:
    n = len(mixed_strategy)
    rng = np.random.default_rng(rng_seed)
    # sample a_{-i}
    sampled_others = []
    for _ in range(br_samples):
        sample = []
        for j in range(n):
            if j == i:
                sample.append(None)
            else:
                sample.append(int(rng.choice(len(ACTION_SET), p=mixed_strategy[j])))
        sampled_others.append(sample)
    best_action = None
    best_score = -1e12
    for aidx in range(len(ACTION_SET)):
        total = 0.0
        for samp in sampled_others:
            a_full = list(samp)
            a_full[i] = aidx
            a_full = tuple(a_full)
            hat = hat_v_i_of_joint_action(i, a_full, ACTION_SET, means)
            b = b_i_of_joint_action(i, a_full, ACTION_SET, counts, delta, n)
            total += (hat + b)
        est = total / float(br_samples)
        if est > best_score:
            best_score = est
            best_action = aidx
    return best_action, best_score

# -----------------------
# Surrogate minimization approximate solver (generalized)
# -----------------------
def surrogate_minimization_approx(n: int,
                                  counts: Dict[Tuple[int,int,int], int],
                                  means: Dict[Tuple[int,int,int], float],
                                  ACTION_SET: Tuple[Tuple[int,...], ...],
                                  delta: float = DELTA,
                                  mc_samples: int = MC_SAMPLES,
                                  br_samples: int = BR_SAMPLES,
                                  max_iters: int = MAX_ITERS,
                                  alpha: float = ALPHA,
                                  rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    phi = [np.ones(len(ACTION_SET)) / len(ACTION_SET) for _ in range(n)]
    # initial surrogate gap estimate
    V_lower, V_upper = estimate_V_bounds_for_all_agents(phi, counts, means, ACTION_SET, delta, mc_samples, rng_seed + 1)
    br_values = np.zeros(n, dtype=float)
    for i in range(n):
        _, br_val = compute_optimistic_best_response_for_agent(i, phi, counts, means, ACTION_SET, delta, br_samples, rng_seed + 10 + i)
        br_values[i] = br_val
    hat_gap = float(np.max(br_values - V_lower))
    best_gap = hat_gap
    best_phi = [p.copy() for p in phi]
    diagnostics = {'gaps':[hat_gap]}

    for it in range(max_iters):
        for i in range(n):
            best_action, _ = compute_optimistic_best_response_for_agent(i, phi, counts, means, ACTION_SET, delta, br_samples, rng_seed + 1000 + it * n + i)
            one_hot = np.zeros(len(ACTION_SET))
            one_hot[best_action] = 1.0
            phi[i] = (1.0 - alpha) * phi[i] + alpha * one_hot
        V_lower, _ = estimate_V_bounds_for_all_agents(phi, counts, means, ACTION_SET, delta, mc_samples, rng_seed + 2000 + it)
        for i in range(n):
            _, br_val = compute_optimistic_best_response_for_agent(i, phi, counts, means, ACTION_SET, delta, br_samples, rng_seed + 3000 + it * n + i)
            br_values[i] = br_val
        hat_gap = float(np.max(br_values - V_lower))
        diagnostics['gaps'].append(hat_gap)
        if hat_gap + 1e-12 < best_gap:
            best_gap = hat_gap
            best_phi = [p.copy() for p in phi]
        if it > 0 and abs(diagnostics['gaps'][-2] - diagnostics['gaps'][-1]) < CONV_TOL:
            break

    # final estimation on best_phi
    V_lower, V_upper = estimate_V_bounds_for_all_agents(best_phi, counts, means, ACTION_SET, delta, mc_samples, rng_seed + 99999)
    final_br_values = np.zeros(n, dtype=float)
    for i in range(n):
        _, br_val = compute_optimistic_best_response_for_agent(i, best_phi, counts, means, ACTION_SET, delta, br_samples, rng_seed + 50000 + i)
        final_br_values[i] = br_val
    final_gap = float(np.max(final_br_values - V_lower))
    return best_phi, final_gap, diagnostics

# -----------------------
# Single experiment run
# -----------------------
def run_single_experiment(n: int, M: int, policy: str, seed: int,
                          K: int, action_set_size: int, utility_model: str):
    rng = np.random.default_rng(seed)
    print(f"Running experiment n={n}, M={M}, policy={policy}, seed={seed} ...")
    print("Generating action set...")
    ACTION_SET = generate_random_action_set(K, action_set_size, rng)
    print("Generating dataset...")
    list_actions, dict_values = generate_offline_dataset(n, M, policy, seed + 7, K, ACTION_SET, utility_model)
    print("Computing empirical estimates...")
    counts, means = compute_empirical_pair_estimates(n, list_actions, dict_values)
    print("Running surrogate minimization approximate solver...")
    phi_out, gap_est, diag = surrogate_minimization_approx(n, counts, means, ACTION_SET,
                                                          delta=DELTA, mc_samples=MC_SAMPLES,
                                                          br_samples=BR_SAMPLES, max_iters=MAX_ITERS,
                                                          alpha=ALPHA, rng_seed=seed + 33)
    print("==============================")
    return gap_est

# -----------------------
# Grid runner for requested plots
# -----------------------
def run_experiment_grid_for_model(utility_model: str,
                                  policy: str = 'random',
                                  n_list: List[int] = [5,10,25,50,75],
                                  K_for_n_plot: int = 5,
                                  K_list_for_Kplot: List[int] = [5,10,15,20,25],
                                  M_values: List[int] = [100,10000,30000,50000,70000,100000],
                                  repeats: int = REPEATS,
                                  action_set_size: int = ACTION_SET_SIZE,
                                  parallel_jobs: int = 1):
    """
    Run two experiment families for a given utility_model:
    - Family A (impact of n): for each n in n_list, vary M over M_values (K fixed to K_for_n_plot).
    - Family B (impact of K for n=25): for each K in K_list_for_Kplot, vary M over M_values (n fixed to 25).
    Returns two result dictionaries mapping (n,M) or (K,M) -> {'mean':..., 'std':..., 'gaps': array}
    """
    # Create list of tasks for family A
    tasks_A = []
    for n in n_list:
        for M in M_values:
            tasks_A.append(('A', n, M, K_for_n_plot))
    # Family B (n fixed to 25)
    n_fixed = 25
    tasks_B = []
    for K in K_list_for_Kplot:
        for M in M_values:
            tasks_B.append(('B', n_fixed, M, K))

    all_tasks = tasks_A + tasks_B

    def run_task(task):
        family, n, M, K = task
        gaps = []
        # each repeat different seed
        for r in range(repeats):
            seed = SEED_BASE + (hash((utility_model, family, n, M, K, r)) % 2_000_000)
            gap = run_single_experiment(n, M, policy, seed, K, action_set_size, utility_model)
            gaps.append(gap)
        gaps = np.array(gaps)
        return (family, n, M, K, gaps)

    if parallel_jobs == 1:
        outputs = [run_task(t) for t in all_tasks]
    else:
        outputs = Parallel(n_jobs=parallel_jobs)(delayed(run_task)(t) for t in all_tasks)

    results_A = {}
    results_B = {}
    for family, n, M, K, gaps in outputs:
        info = {'gaps': gaps, 'mean': float(np.mean(gaps)), 'std': float(np.std(gaps, ddof=1))}
        if family == 'A':
            results_A[(n, M)] = info
        else:
            results_B[(K, M)] = info
    return results_A, results_B

# -----------------------
# Plotting utilities
# -----------------------
def plot_impact_of_n(results_A: Dict, n_list: List[int], M_values: List[int], utility_model: str, outname: str, policy: str):
    plt.figure(figsize=(10,6))
    for n in n_list:
        means = [results_A[(n, M)]['mean'] for M in M_values]
        stds = [results_A[(n, M)]['std'] for M in M_values]
        plt.plot(M_values, means, marker='o', label=f'n={n}')
        plt.fill_between(M_values, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.15)
    plt.xscale('log')
    plt.xlabel('Dataset size M (log scale)')
    plt.ylabel('Duality gap (surrogate estimate)')
    plt.title(f'Impact of number of agents n (utility model={utility_model}, policy={policy})')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, outname)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved plot {path}")

def plot_impact_of_K(results_B: Dict, K_list: List[int], M_values: List[int], utility_model: str, outname: str, policy: str):
    plt.figure(figsize=(10,6))
    for K in K_list:
        means = [results_B[(K, M)]['mean'] for M in M_values]
        stds = [results_B[(K, M)]['std'] for M in M_values]
        plt.plot(M_values, means, marker='o', label=f'K={K}')
        plt.fill_between(M_values, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.15)
    plt.xscale('log')
    plt.xlabel('Dataset size M (log scale)')
    plt.ylabel('Duality gap (surrogate estimate)')
    plt.title(f'Impact of number of coalitions k (n=25) (utility model={utility_model}, policy={policy})')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, outname)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved plot {path}")

# -----------------------
# Master runner over all utility models
# -----------------------
def run_all_models(parallel_jobs: int = 1):
    # parameters requested by user
    M_values = [100, 5000, 10000, 20000, 30000]
    n_list = [5, 10, 25, 50, 75]
    K_for_n_plot = 5  # for the impact-of-n plot we fix K=5
    K_list_for_Kplot = [5, 10, 15, 20, 25]

    for utility_model in UTILITY_MODELS:
        print("\n\n==============================")
        print(f"Running experiments for utility model: {utility_model}")
        print("==============================")
        start_time = time.time()
        results_A, results_B = run_experiment_grid_for_model(utility_model=utility_model,
                                                            policy='random',  # you can change to 'coalitionSize' if desired
                                                            n_list=n_list,
                                                            K_for_n_plot=K_for_n_plot,
                                                            K_list_for_Kplot=K_list_for_Kplot,
                                                            M_values=M_values,
                                                            repeats=REPEATS,
                                                            action_set_size=ACTION_SET_SIZE,
                                                            parallel_jobs=parallel_jobs)
        elapsed = time.time() - start_time
        print(f"Finished model {utility_model} in {elapsed:.1f}s")

        # Plot and save
        plot_impact_of_n(results_A, n_list, M_values, utility_model, outname=f"impact_n_{utility_model}.png")
        plot_impact_of_K(results_B, K_list_for_Kplot, M_values, utility_model, outname=f"impact_K_n25_{utility_model}.png")

        # Save raw results
        savepath = os.path.join(OUT_DIR, f"raw_results_{utility_model}.pkl")
        with open(savepath, "wb") as f:
            pickle.dump({'results_A': results_A, 'results_B': results_B}, f)
        print(f"Saved raw results to {savepath}")

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    # NOTE: this will run the full experiment for all four utility models and
    # produce 2 plots per model (8 plots). This is computationally heavy.
    # To run faster for debugging, reduce REPEATS, MC_SAMPLES, BR_SAMPLES, or run for a single model.
    import argparse
    parser = argparse.ArgumentParser(description="Run POCF semi-bandit empirical experiments across utility models.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel jobs for grid tasks (joblib).")
    parser.add_argument("--single-model", type=str, default=None, choices=UTILITY_MODELS + [None], help="Run only a single utility model (faster).")
    args = parser.parse_args()
    policy='random'
    if args.single_model is not None:
        # run only one model (useful for debugging / quicker runs)
        model = args.single_model
        print(f"Running single model: {model}")
        start = time.time()
        results_A, results_B = run_experiment_grid_for_model(utility_model=model,
                                                            policy='random',
                                                            n_list=[5,10,25,50,75],
                                                            K_for_n_plot=5,
                                                            K_list_for_Kplot=[5,10,15,20,25],
                                                            M_values=[100, 5000, 10000, 20000, 30000],
                                                            repeats=REPEATS,
                                                            action_set_size=ACTION_SET_SIZE,
                                                            parallel_jobs=args.parallel)
        plot_impact_of_n(results_A, [5,10,25,50,75], [100, 5000, 10000, 20000, 30000], model, outname=f"impact_n_{model}_{policy}.png", policy=policy)
        plot_impact_of_K(results_B, [5,10,15,20,25], [100, 5000, 10000, 20000, 30000], model, outname=f"impact_K_n25_{model}_{policy}.png", policy=policy)
        with open(os.path.join(OUT_DIR, f"raw_results_{model}.pkl"), "wb") as f:
            pickle.dump({'results_A': results_A, 'results_B': results_B}, f)
        print("Done. Time:", time.time() - start)
    else:
        run_all_models(parallel_jobs=args.parallel)
