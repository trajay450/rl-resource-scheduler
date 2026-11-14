#!/usr/bin/env python3
"""Reproduce every number and figure in the README.

    python experiments/run_experiment.py   (~20s on a laptop CPU)
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlscheduler.agents import (
    QLearningAgent,
    run_greedy_most_capacity_policy,
    run_random_policy,
    run_round_robin_policy,
)
from rlscheduler.env import JOB_SIZES, SchedulerEnv

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CAPACITIES = (3, 5, 8)
N_EPISODES = 900
STEPS_PER_EPISODE = 200
EVAL_STEPS = 2000
EVAL_SEEDS = list(range(10))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    agent = QLearningAgent(capacities=CAPACITIES, seed=0, alpha=0.2, epsilon_decay_episodes=600)
    training_rewards = agent.train(N_EPISODES, STEPS_PER_EPISODE, env_seed=0)

    rng = np.random.default_rng(99)
    policies = {
        "random": lambda env: run_random_policy(env, EVAL_STEPS, rng),
        "round_robin": lambda env: run_round_robin_policy(env, EVAL_STEPS),
        "greedy_most_capacity": lambda env: run_greedy_most_capacity_policy(env, EVAL_STEPS),
        "q_learning": lambda env: agent.run_greedy(env, EVAL_STEPS),
    }

    summary = {}
    for name, fn in policies.items():
        totals = []
        for seed in EVAL_SEEDS:
            env = SchedulerEnv(capacities=CAPACITIES, seed=1000 + seed)
            totals.append(fn(env))
        summary[name] = {"mean_total_reward": float(np.mean(totals)), "std_total_reward": float(np.std(totals))}

    best_baseline = max(
        (v["mean_total_reward"] for k, v in summary.items() if k != "q_learning")
    )
    summary["q_learning"]["improvement_over_best_baseline_pct"] = float(
        100 * (summary["q_learning"]["mean_total_reward"] - best_baseline) / best_baseline
    )

    # placement diagnostics: where does each policy send small/large jobs?
    def track(env, steps, action_fn, placements):
        for _ in range(steps):
            job = env.current_job_size
            action = action_fn(env)
            if action < env.n_servers:
                placements[action][job] += 1
            env.step(action)

    env_g = SchedulerEnv(capacities=CAPACITIES, seed=2000)
    placements_greedy = {i: Counter() for i in range(len(CAPACITIES))}
    track(env_g, EVAL_STEPS, lambda e: max((a for a in e.valid_actions() if a < e.n_servers), key=lambda a: e.free[a], default=e.n_servers), placements_greedy)

    env_q = SchedulerEnv(capacities=CAPACITIES, seed=2000)
    placements_q = {i: Counter() for i in range(len(CAPACITIES))}
    track(env_q, EVAL_STEPS, lambda e: agent.act(e.state(), e.valid_actions(), epsilon=0.0), placements_q)

    metrics = {
        "config": {
            "server_capacities": list(CAPACITIES),
            "job_sizes": JOB_SIZES,
            "n_training_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_steps_per_run": EVAL_STEPS,
            "eval_seeds": len(EVAL_SEEDS),
        },
        "policy_comparison": summary,
        "placement_distribution": {
            "greedy_most_capacity": {str(k): dict(v) for k, v in placements_greedy.items()},
            "q_learning": {str(k): dict(v) for k, v in placements_q.items()},
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # plot 1: training curve
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    window = 20
    smoothed = np.convolve(training_rewards, np.ones(window) / window, mode="valid")
    ax.plot(training_rewards, alpha=0.25, color="#378ADD", linewidth=0.8, label="per-episode reward")
    ax.plot(range(window - 1, len(training_rewards)), smoothed, color="#042C53", linewidth=1.8, label=f"{window}-episode moving average")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Total episode reward")
    ax.set_title("Q-learning convergence")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curve.png", dpi=160)
    plt.close(fig)

    # plot 2: policy comparison bar chart
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    names = list(policies.keys())
    means = [summary[n]["mean_total_reward"] for n in names]
    stds = [summary[n]["std_total_reward"] for n in names]
    colors = ["#888780", "#BA7517", "#D85A30", "#639922"]
    ax.bar(names, means, yerr=stds, capsize=4, color=colors)
    ax.set_ylabel(f"Mean total reward over {EVAL_STEPS} steps ({len(EVAL_SEEDS)} seeds)")
    ax.set_title("Learned policy vs. fixed heuristics")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "policy_comparison.png", dpi=160)
    plt.close(fig)

    # plot 3: where each policy sends small vs. large jobs (big server = index 2)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, (name, placements) in zip(axes, [("Greedy (most free capacity)", placements_greedy), ("Q-learning", placements_q)]):
        x = np.arange(len(CAPACITIES))
        width = 0.25
        for i, size in enumerate(JOB_SIZES):
            counts = [placements[s].get(size, 0) for s in range(len(CAPACITIES))]
            ax.bar(x + (i - 1) * width, counts, width, label=f"size {size}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"server {i}\n(cap={c})" for i, c in enumerate(CAPACITIES)])
        ax.set_title(name)
    axes[0].set_ylabel("Jobs placed")
    axes[1].legend(fontsize=8.5, title="job size")
    fig.suptitle("Where does each policy send small vs. large jobs?")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "placement_distribution.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
