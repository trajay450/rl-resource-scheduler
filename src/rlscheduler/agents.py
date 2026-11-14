"""The learned agent (tabular Q-learning) and three fixed-policy baselines."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rlscheduler.env import DEFAULT_CAPACITIES, SchedulerEnv


@dataclass
class QLearningAgent:
    capacities: tuple[int, ...] = DEFAULT_CAPACITIES
    alpha: float = 0.15
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 300
    seed: int = 0
    q_: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.n_servers = len(self.capacities)

    def _q(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q_:
            self.q_[state] = np.zeros(self.n_servers + 1)
        return self.q_[state]

    def act(self, state: tuple[int, ...], valid_actions: list[int], epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return int(self._rng.choice(valid_actions))
        qvals = self._q(state)
        masked = np.full_like(qvals, -np.inf)
        masked[valid_actions] = qvals[valid_actions]
        return int(np.argmax(masked))

    def update(self, state, action, reward, next_state, next_valid_actions) -> None:
        q_sa = self._q(state)[action]
        next_q = self._q(next_state)
        best_next = max(next_q[a] for a in next_valid_actions)
        td_target = reward + self.gamma * best_next
        self._q(state)[action] += self.alpha * (td_target - q_sa)

    def epsilon_for_episode(self, episode: int) -> float:
        frac = min(episode / self.epsilon_decay_episodes, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def train(self, n_episodes: int, steps_per_episode: int, env_seed: int = 0) -> list[float]:
        episode_rewards = []
        for ep in range(n_episodes):
            env = SchedulerEnv(capacities=self.capacities, seed=env_seed * 10_000 + ep)
            state = env.reset()
            epsilon = self.epsilon_for_episode(ep)
            total_reward = 0.0
            for _ in range(steps_per_episode):
                valid = env.valid_actions()
                action = self.act(state, valid, epsilon)
                next_state, reward, _ = env.step(action)
                next_valid = env.valid_actions()
                self.update(state, action, reward, next_state, next_valid)
                state = next_state
                total_reward += reward
            episode_rewards.append(total_reward)
        return episode_rewards

    def run_greedy(self, env: SchedulerEnv, steps: int) -> float:
        state = env.state()
        total = 0.0
        for _ in range(steps):
            valid = env.valid_actions()
            action = self.act(state, valid, epsilon=0.0)
            state, reward, _ = env.step(action)
            total += reward
        return total


# ---- fixed-policy baselines ----

def run_random_policy(env: SchedulerEnv, steps: int, rng: np.random.Generator) -> float:
    total = 0.0
    for _ in range(steps):
        valid = env.valid_actions()
        action = int(rng.choice(valid))
        _, reward, _ = env.step(action)
        total += reward
    return total


def run_round_robin_policy(env: SchedulerEnv, steps: int) -> float:
    total = 0.0
    next_server = 0
    for _ in range(steps):
        valid = env.valid_actions()
        action = next_server if next_server in valid else env.n_servers  # reject if it doesn't fit
        next_server = (next_server + 1) % env.n_servers
        _, reward, _ = env.step(action)
        total += reward
    return total


def run_greedy_most_capacity_policy(env: SchedulerEnv, steps: int) -> float:
    """A reasonable non-learned heuristic: always place the job on whichever
    server currently has the most free capacity (load-balancing), rejecting
    only if nothing fits."""
    total = 0.0
    for _ in range(steps):
        valid = [a for a in env.valid_actions() if a < env.n_servers]
        if valid:
            action = max(valid, key=lambda a: env.free[a])
        else:
            action = env.n_servers
        _, reward, _ = env.step(action)
        total += reward
    return total
