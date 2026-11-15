import collections.abc
import contextlib
import pathlib
import typing

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from experiments import config, runs

DEFAULT_EPISODE_STEPS = 200


class DiscreteMDP(typing.Protocol):
    """The finite-state reset interface exposed by Gymnasium's toy environments."""

    initial_state_distrib: npt.NDArray[np.float64]
    P: dict[int, dict[int, list[tuple[float, int, float, bool]]]]


def randomize_starts(env: gym.Env) -> None:
    """Sample training resets uniformly from reachable nonterminal states."""
    base = typing.cast(DiscreteMDP, env.unwrapped)
    reachable = {int(state) for state in np.flatnonzero(base.initial_state_distrib)}
    pending = sorted(reachable)
    while pending:
        state = pending.pop()
        for choices in base.P[state].values():
            for probability, next_state, _, terminal in choices:
                if probability > 0 and not terminal and next_state not in reachable:
                    reachable.add(next_state)
                    pending.append(next_state)
    probabilities = np.zeros_like(base.initial_state_distrib)
    probabilities[sorted(reachable)] = 1.0 / len(reachable)
    base.initial_state_distrib = probabilities


@contextlib.contextmanager
def open(
    settings: config.TrainConfig | runs.TrainingSettings,
    *,
    max_episode_steps: int | None = None,
    training: bool = False,
    video_dir: pathlib.Path | None = None,
    watch: bool = False,
) -> collections.abc.Generator[gym.Env]:
    """Create, wrap, and close one environment."""
    env = gym.make(
        settings.env,
        max_episode_steps=(
            max_episode_steps or gym.spec(settings.env).max_episode_steps or DEFAULT_EPISODE_STEPS
        ),
        render_mode="rgb_array" if video_dir is not None or watch else None,
    )
    try:
        if isinstance(env.observation_space, gym.spaces.Discrete):
            if training:
                randomize_starts(env)
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.DtypeObservation(env, dtype=np.float32)
        if settings.scale_observations:
            space = typing.cast(gym.spaces.Box, env.observation_space)
            bounded = np.isfinite(space.low) & np.isfinite(space.high)
            env = gym.wrappers.RescaleObservation(
                env,
                min_obs=np.where(bounded, -1.0, space.low),
                max_obs=np.where(bounded, 1.0, space.high),
            )
        if video_dir is not None:
            env = gym.wrappers.RecordVideo(
                env,
                str(video_dir),
                episode_trigger=lambda _: True,
            )
        if watch:
            env = gym.wrappers.HumanRendering(env)
        yield env
    finally:
        env.close()


@contextlib.contextmanager
def open_many(
    cfg: config.TrainConfig,
    *,
    count: int,
) -> collections.abc.Generator[list[gym.Env]]:
    """Create training environments and close them together."""
    with contextlib.ExitStack() as stack:
        yield [
            stack.enter_context(
                open(
                    cfg,
                    max_episode_steps=cfg.max_episode_steps,
                    training=True,
                )
            )
            for _ in range(count)
        ]
