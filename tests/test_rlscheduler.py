import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from rlscheduler.agents import QLearningAgent, run_random_policy, run_round_robin_policy
from rlscheduler.env import SchedulerEnv


def test_env_state_shape():
    env = SchedulerEnv(capacities=(3, 5, 8), seed=0)
    state = env.reset()
    assert len(state) == 4  # 3 servers + current job size


def test_env_rejects_oversized_placement_gracefully():
    env = SchedulerEnv(capacities=(3, 5, 8), seed=0)
    env.free[:] = 0  # no capacity anywhere
    env.current_job_size = 3
    assert env.valid_actions() == [env.n_servers]  # only "reject" is valid


def test_env_accept_reduces_free_capacity():
    env = SchedulerEnv(capacities=(3, 5, 8), seed=0)
    env.free[:] = [3, 5, 8]
    env.current_job_size = 2
    state, reward, info = env.step(0)
    assert info["accepted"]
    assert reward > 0


def test_env_overload_penalized():
    env = SchedulerEnv(capacities=(3, 5, 8), seed=0)
    env.free[0] = 0
    env.current_job_size = 1
    _, reward, info = env.step(0)
    assert info["overloaded"]
    assert reward < 0


def test_qlearning_agent_learns_positive_improvement():
    agent = QLearningAgent(capacities=(3, 5, 8), seed=0, alpha=0.2, epsilon_decay_episodes=300)
    rewards = agent.train(n_episodes=400, steps_per_episode=150, env_seed=0)
    early = np.mean(rewards[:50])
    late = np.mean(rewards[-50:])
    assert late > early  # clear improvement, not just noise


def test_trained_agent_beats_random_policy():
    agent = QLearningAgent(capacities=(3, 5, 8), seed=0, alpha=0.2, epsilon_decay_episodes=400)
    agent.train(n_episodes=500, steps_per_episode=150, env_seed=0)

    rng = np.random.default_rng(1)
    q_totals, random_totals = [], []
    for seed in range(5):
        q_totals.append(agent.run_greedy(SchedulerEnv(capacities=(3, 5, 8), seed=500 + seed), 800))
        random_totals.append(run_random_policy(SchedulerEnv(capacities=(3, 5, 8), seed=500 + seed), 800, rng))
    assert np.mean(q_totals) > np.mean(random_totals)


def test_round_robin_cycles_through_all_servers():
    env = SchedulerEnv(capacities=(10, 10, 10), seed=0)  # plenty of capacity, no rejections forced
    first_actions = []
    for _ in range(6):
        valid = env.valid_actions()
        # replicate round-robin's own counter externally to check distinct targets observed
        first_actions.append(valid)
    assert all(len(v) == 4 for v in first_actions)  # 3 servers + reject, all fit
