from __future__ import annotations

import collections.abc
import dataclasses
import typing

import gymnasium as gym
import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from a2c import agent as agents


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Transition:
    """One step per environment; fields are frozen, array contents are mutable."""

    # B = batch size, D = observation features.
    states: npt.NDArray[np.float32]  # (B, D)
    actions: npt.NDArray[np.int64]  # (B,)
    next_states: npt.NDArray[np.float32]  # (B, D), before reset
    rewards: npt.NDArray[np.float32]  # (B,)
    terminated: npt.NDArray[np.bool_]  # (B,), true terminal states
    truncated: npt.NDArray[np.bool_]  # (B,), external episode endings


class GenerateTransitions:
    """Collect one step per environment, retaining observations before any reset."""

    def __init__(
        self,
        envs: collections.abc.Sequence[gym.Env],
        agent: agents.PolicyAgent,
        seed: int,
    ) -> None:
        self.envs = envs
        self.agent = agent
        self.states = np.stack(
            [env.reset(seed=(seed + i) % 2**32)[0] for i, env in enumerate(envs)],
            dtype=np.float32,
        )

    def __iter__(self) -> collections.abc.Iterator[Transition]:
        while True:
            actions = self.agent.actions(self.states)
            next_states: list[npt.ArrayLike] = []
            rewards: list[typing.SupportsFloat] = []
            terminated: list[bool] = []
            truncated: list[bool] = []
            continuation: list[npt.ArrayLike] = []
            for env, action in zip(self.envs, actions):
                state, reward, terminal, timeout, _ = env.step(int(action))
                next_states.append(np.array(state, copy=True))
                rewards.append(reward)
                terminated.append(terminal)
                truncated.append(timeout)
                continuation.append(env.reset()[0] if terminal or timeout else state)

            yield Transition(
                self.states.copy(),
                np.asarray(actions, dtype=np.int64),
                np.stack(next_states, dtype=np.float32),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(terminated, dtype=np.bool_),
                np.asarray(truncated, dtype=np.bool_),
            )
            self.states = np.stack(continuation, dtype=np.float32)
