import pathlib
import shutil
import typing

import gymnasium as gym
import moviepy
import numpy as np
import pydantic
import pydantic_settings

from experiments import config, runs

EVALUATION_SEED = 1000


class Evaluation(pydantic.BaseModel):
    """Greedy evaluation over consecutive episode seeds."""

    model_config = pydantic.ConfigDict(allow_inf_nan=False)

    run_name: str
    env: config.Environment
    seeds: list[pydantic.NonNegativeInt] = pydantic.Field(min_length=1)
    returns: list[float] = pydantic.Field(min_length=1)
    lengths: list[pydantic.PositiveInt] = pydantic.Field(min_length=1)

    @pydantic.model_validator(mode="after")
    def consistent_episodes(self) -> typing.Self:
        if not len(self.seeds) == len(self.returns) == len(self.lengths):
            raise ValueError("Seeds, returns, and lengths must have equal sizes")
        if self.seeds != list(range(self.seeds[0], self.seeds[0] + len(self.seeds))):
            raise ValueError("Evaluation seeds must be consecutive")
        return self

    @pydantic.computed_field
    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns))

    @pydantic.computed_field
    @property
    def std_return(self) -> float:
        return float(np.std(self.returns))


class Score(pydantic.BaseModel):
    run: runs.Run
    evaluation: Evaluation
    target: float


class ScoresConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        cli_kebab_case=True, cli_enforce_required=True
    )

    run_names: pydantic_settings.CliPositionalArg[list[str]]
    output: pathlib.Path = config.RUNS_DIR / "scores"


def write_scores(run_names: list[str], output: pathlib.Path) -> pathlib.Path:
    """Publish selected runs and their demos as a Markdown scorecard."""
    scores: list[Score] = []
    for name in run_names:
        run = runs.Run.load(name)
        report = Evaluation.model_validate_json(
            (run.path / "analysis" / "evaluation.json").read_text()
        )
        if report.run_name != name or report.env != run.settings.env:
            raise ValueError(f"Evaluation does not match run: {name}")
        target = -13.0 if report.env == "CliffWalking-v1" else gym.spec(report.env).reward_threshold
        if target is None:
            raise ValueError(f"No score threshold registered for {report.env}")
        scores.append(Score(run=run, evaluation=report, target=target))
    if not scores or len({score.evaluation.env for score in scores}) != len(scores):
        raise ValueError("Select exactly one run per environment")
    if any(score.evaluation.seeds != scores[0].evaluation.seeds for score in scores):
        raise ValueError("Selected runs must use the same evaluation seeds")
    scores.sort(key=lambda score: score.evaluation.env)
    seeds = scores[0].evaluation.seeds
    lines = [
        "# Reference scores",
        "",
        f"{len(seeds)} episodes per environment · greedy actions · seeds {seeds[0]}–{seeds[-1]}.",
        "Evaluation uses standard starts and episode limits; CliffWalking uses a 200-step cap.",
        "Higher scores are better.",
        "",
        "| Environment | Target mean ≥ | Achieved mean ± std | Target met |",
        "| :-- | --: | --: | :--: |",
    ]
    for score in scores:
        report = score.evaluation
        met = "✓" if report.mean_return >= score.target else "—"
        digits = 3 if abs(score.target) < 1 else 2
        lines.append(
            f"| {report.env} | {score.target:g} | "
            f"{report.mean_return:.{digits}f} ± {report.std_return:.{digits}f} | {met} |"
        )
    if any(score.evaluation.env == "FrozenLake-v1" for score in scores):
        lines += [
            "",
            "FrozenLake's mean return is its success rate. Its slippery dynamics can cause",
            "individual failures, including the fixed first-seed demo, even with an optimal policy.",
        ]
    lines += [
        "",
        "Targets come from Gymnasium's registered reward thresholds, except CliffWalking:",
        "its project target of −13 is the optimal 13-step safe route (no official threshold).",
        "The 200-step cap prevents an unsuccessful CliffWalking policy looping forever. These are selected",
        "single-seed examples, not averages across independent training runs.",
        "",
        "## Training settings",
        "",
        "| Environment | Seed | Steps | Critic LR | Actor LR | Gamma | Entropy | Adam ε | Scaling | Episode limit |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: | :--: | --: |",
    ]
    for score in scores:
        run = score.run
        s = run.settings
        steps = ((s.total_steps + run.n_envs - 1) // run.n_envs) * run.n_envs
        limit = str(s.max_episode_steps) if s.max_episode_steps is not None else "Standard"
        lines.append(
            f"| {s.env} | {s.seed} | {steps:,} | {s.lr_value:g} | {s.lr_policy:g} | "
            f"{s.gamma:g} | {s.beta:g} | {s.adam_eps:g} | {'Yes' if s.scale_observations else 'No'} | {limit} |"
        )
    lines += [
        "",
        "The reference script uses 64 training environments, CPU execution, one computation thread, and locked dependencies.",
        "Training starts and limits may differ from evaluation. See `scripts/reproduce.sh`.",
        "All runs use two 64-unit Tanh layers with orthogonal initialization.",
        "Discrete-state training uses exploring starts; evaluation always uses normal starts.",
        "Exact results can differ across platforms.",
        "",
        "## Runs and previews",
        "",
    ]
    output.mkdir(parents=True, exist_ok=True)
    for score in scores:
        run = score.run
        slug = run.settings.env.split("-v")[0].lower()
        analysis = run.path / "analysis"
        lines += [f"### {run.settings.env}", "", f"Run: `{run.run_name}`.", ""]
        if (analysis / "training.png").exists():
            shutil.copy2(analysis / "training.png", output / f"{slug}-training.png")

        if (analysis / "episode.mp4").exists():
            with moviepy.VideoFileClip(str(analysis / "episode.mp4")) as clip:
                preview = typing.cast(moviepy.VideoClip, clip.resized(width=320))
                preview.write_gif(str(output / f"{slug}.gif"), fps=15, logger=None)
            lines += [f"![{run.settings.env}: episode seed {seeds[0]}]({slug}.gif)", ""]
        if (output / f"{slug}-training.png").exists():
            lines += [f"![{run.settings.env} training]({slug}-training.png)", ""]
    path = output / "scores.md"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    return path


if __name__ == "__main__":
    options = pydantic_settings.CliApp.run(ScoresConfig)
    print(f"Saved {write_scores(options.run_names, options.output)}")
