import collections
import csv
import pathlib
import tempfile

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings

from a2c import agent
from experiments import config, envs, runs, scores, utils


def episode(policy: agent.PolicyAgent, env: gym.Env, seed: int) -> tuple[float, int]:
    """Play one episode without learning."""
    state, _ = env.reset(seed=seed)
    score, length = 0.0, 0
    while True:
        action = policy.actions([state], stochastic=False)[0]
        state, reward, terminated, truncated, _ = env.step(int(action))
        score += float(reward)
        length += 1
        if terminated or truncated:
            return score, length


def evaluate(policy: agent.PolicyAgent, run: runs.Run, episodes: int) -> scores.Evaluation:
    """Evaluate every run on the same seed sequence."""
    seeds = list(range(scores.EVALUATION_SEED, scores.EVALUATION_SEED + episodes))
    returns, lengths = [], []
    with envs.open(run.settings) as env:
        for seed in seeds:
            score, length = episode(policy, env, seed)
            returns.append(score)
            lengths.append(length)
    return scores.Evaluation(
        run_name=run.run_name,
        env=run.settings.env,
        seeds=seeds,
        returns=returns,
        lengths=lengths,
    )


def plot_training(run: runs.Run, output: pathlib.Path) -> None:
    """Plot episode returns and their rolling mean."""
    with (run.path / "metrics.csv").open() as file:
        metrics = [runs.Episode.model_validate(row) for row in csv.DictReader(file)]
    history: collections.deque[float] = collections.deque(maxlen=20)
    smooth = []
    for metric in metrics:
        history.append(metric.episode_return)
        smooth.append(float(np.mean(history)))
    steps = [metric.env_steps for metric in metrics]
    fig, ax = plt.subplots()
    if metrics:
        ax.scatter(
            steps, [metric.episode_return for metric in metrics], s=8, alpha=0.2, label="Episodes"
        )
        ax.plot(steps, smooth, label="20-episode mean (all environments)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No completed training episodes", ha="center", transform=ax.transAxes)
    ax.set(title=run.run_name, xlabel="Total environment steps", ylabel="Episode return")
    fig.savefig(output / "training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def demo_episode(
    policy: agent.PolicyAgent,
    run: runs.Run,
    output: pathlib.Path,
    *,
    record: bool,
    watch: bool,
) -> None:
    """Watch and/or record the same single episode."""
    with tempfile.TemporaryDirectory(dir=output) as directory:
        with envs.open(
            run.settings,
            video_dir=pathlib.Path(directory) / "video" if record else None,
            watch=watch,
        ) as env:
            episode(policy, env, scores.EVALUATION_SEED)

        if record:
            next(pathlib.Path(directory).rglob("*.mp4")).replace(output / "episode.mp4")


def analyse(cfg: config.AnalyseConfig) -> scores.Evaluation:
    """Evaluate, plot, and optionally record a named run."""
    run = runs.Run.load(cfg.run_name)
    policy = run.load_weights(device=utils.choose_device(cfg.device))
    policy.policy_net.eval()
    policy.value_net.eval()

    output = run.path / "analysis"
    output.mkdir(exist_ok=True)
    report = evaluate(policy, run, cfg.episodes)
    temporary = output / "evaluation.tmp"
    temporary.write_text(report.model_dump_json(indent=2) + "\n")
    temporary.replace(output / "evaluation.json")
    plot_training(run, output)

    if cfg.record or cfg.watch:
        demo_episode(policy, run, output, record=cfg.record, watch=cfg.watch)

    print(
        f"{run.run_name}: {report.mean_return:.2f} ± {report.std_return:.2f} "
        f"({cfg.episodes} episodes).\nSaved {output}"
    )
    return report


if __name__ == "__main__":
    analyse(pydantic_settings.CliApp.run(config.AnalyseConfig))
