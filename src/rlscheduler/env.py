"""A small, fully custom (no gym dependency) multi-server job-scheduling
environment: jobs of varying size and value arrive one at a time; the
agent assigns each to one of K servers (or rejects it); server capacity
is finite and regenerates stochastically as jobs complete. This is a
simplified, tabular-friendly stand-in for cloud resource allocation,
network slicing, or hospital bed/operating-room scheduling — anywhere a
system must decide, online, whether and where to place an arriving
request under capacity constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

JOB_SIZES = [1, 2, 3]          # small, medium, large
JOB_VALUES = [5, 12, 45]       # reward for successfully serving each size --
                                # large jobs are disproportionately valuable,
                                # so blocking one is costly: a policy that
                                # reserves big-server headroom for them (by
                                # routing small jobs elsewhere even when the
                                # big server looks "most free" right now)
                                # should out-earn one that doesn't.
OVERLOAD_PENALTY = -20.0       # tried to place a job that doesn't fit
REJECT_PENALTY = -2.0          # turned a job away (opportunity cost)
REGEN_PROB = 0.3               # chance each server regains 1 capacity unit per step


DEFAULT_CAPACITIES = (3, 5, 8)  # heterogeneous: a small, medium and large server


@dataclass
class SchedulerEnv:
    n_servers: int = 3
    capacities: tuple[int, ...] = DEFAULT_CAPACITIES
    seed: int = 0
    _rng: np.random.Generator = field(default=None, init=False, repr=False)
    free: np.ndarray = field(default=None, init=False)
    current_job_size: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.n_servers = len(self.capacities)
        self.capacities = np.array(self.capacities, dtype=int)
        self.reset()

    def reset(self) -> tuple[int, ...]:
        self.free = self.capacities.copy()
        self.current_job_size = int(self._rng.choice(JOB_SIZES))
        return self.state()

    def state(self) -> tuple[int, ...]:
        """(free_0, free_1, ..., free_{K-1}, job_size)."""
        return tuple(int(f) for f in self.free) + (self.current_job_size,)

    @property
    def n_actions(self) -> int:
        return self.n_servers + 1  # one per server, plus "reject"

    def valid_actions(self) -> list[int]:
        valid = [i for i in range(self.n_servers) if self.free[i] >= self.current_job_size]
        valid.append(self.n_servers)  # reject is always valid
        return valid

    def step(self, action: int) -> tuple[tuple[int, ...], float, dict]:
        job_idx = JOB_SIZES.index(self.current_job_size)
        info = {"accepted": False, "overloaded": False, "rejected": False}

        if action == self.n_servers:
            reward = REJECT_PENALTY
            info["rejected"] = True
        elif self.free[action] >= self.current_job_size:
            self.free[action] -= self.current_job_size
            reward = float(JOB_VALUES[job_idx])
            info["accepted"] = True
        else:
            reward = OVERLOAD_PENALTY
            info["overloaded"] = True

        # capacity regeneration (jobs completing and freeing resources)
        for i in range(self.n_servers):
            if self.free[i] < self.capacities[i] and self._rng.random() < REGEN_PROB:
                self.free[i] += 1

        self.current_job_size = int(self._rng.choice(JOB_SIZES))
        return self.state(), reward, info
