import collections
import typing

import gymnasium as gym
import numpy as np
import pydantic_settings
import torch

from a2c import agent, transitions
from experiments import config, envs, runs, utils

N_ENVS = 64
LOG_INTERVAL = 100


def train(cfg: config.TrainConfig) -> runs.Run:
    """Train online and save weights once at completion."""
    torch.manual_seed(cfg.seed)

    with envs.open_many(cfg, count=N_ENVS) as environments:
        env = environments[0]

        observation_size = typing.cast(gym.spaces.Box, env.observation_space).shape[0]
        n_actions = int(typing.cast(gym.spaces.Discrete, env.action_space).n)

        policy = agent.PolicyAgent(
            observation_size,
            n_actions,
            lr_value=cfg.lr_value,
            lr_policy=cfg.lr_policy,
            gamma=cfg.gamma,
            beta=cfg.beta,
            device=utils.choose_device(cfg.device),
            adam_eps=cfg.adam_eps,
        )
        run = runs.Run.create(cfg, policy, N_ENVS)
        print(f"Run: {run.path}", flush=True)
        returns = np.zeros(N_ENVS)
        lengths = np.zeros(N_ENVS, dtype=int)
        recent: collections.deque[float] = collections.deque(maxlen=100)
        for update, transition in enumerate(
            transitions.GenerateTransitions(environments, policy, cfg.seed),
            start=1,
        ):
            policy.update(transition)
            steps = update * N_ENVS
            returns += transition.rewards
            lengths += 1
            for index in np.flatnonzero(transition.terminated | transition.truncated):
                recent.append(float(returns[index]))
                run.log(
                    runs.Episode(
                        env_steps=steps,
                        env_index=int(index),
                        episode_return=float(returns[index]),
                        episode_length=int(lengths[index]),
                    )
                )
                returns[index], lengths[index] = 0, 0
            if update % LOG_INTERVAL == 0:
                score = f"{np.mean(recent):.2f}" if recent else "no completed episodes"
                print(f"steps={steps} mean_return_100={score}", flush=True)
            if steps >= cfg.total_steps:
                break
        run.save_weights(policy)
    return run


if __name__ == "__main__":
    train(pydantic_settings.CliApp.run(config.TrainConfig))
