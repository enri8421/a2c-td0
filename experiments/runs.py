import csv
import pathlib
import typing

import pydantic
import torch

from a2c import agent
from experiments import config as cli


class TrainingSettings(pydantic.BaseModel):
    """Training settings without command-line or hardware details."""

    model_config = pydantic.ConfigDict(from_attributes=True, frozen=True)

    env: cli.Environment
    seed: int
    total_steps: int
    scale_observations: bool
    adam_eps: float
    max_episode_steps: int | None

    lr_value: float
    lr_policy: float
    gamma: float
    beta: float


class Weights(pydantic.BaseModel):
    """Typed state dictionaries at the PyTorch serialization boundary."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    policy: dict[str, torch.Tensor]
    value: dict[str, torch.Tensor]


class Episode(pydantic.BaseModel):
    env_steps: int
    env_index: int
    episode_return: float
    episode_length: int


class Run(pydantic.BaseModel):
    """A run's training settings and network dimensions."""

    run_name: str = pydantic.Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
    settings: TrainingSettings

    observation_size: int = pydantic.Field(gt=0)
    n_actions: int = pydantic.Field(gt=0)
    n_envs: int = pydantic.Field(gt=0)

    @property
    def path(self) -> pathlib.Path:
        return cli.RUNS_DIR / self.run_name

    @classmethod
    def create(
        cls,
        cfg: cli.TrainConfig,
        policy: agent.PolicyAgent,
        n_envs: int,
    ) -> typing.Self:
        """Create a run without overwriting an existing folder."""
        run = cls(
            run_name=cfg.run_name,
            settings=TrainingSettings.model_validate(cfg),
            observation_size=policy.observation_size,
            n_actions=policy.n_actions,
            n_envs=n_envs,
        )
        run.path.mkdir(parents=True, exist_ok=False)
        with (run.path / "metrics.csv").open("x", newline="") as file:
            csv.DictWriter(
                file,
                fieldnames=list(Episode.model_fields),
            ).writeheader()

        run.store()
        return run

    def store(self) -> None:
        """Save the run manifest."""
        temporary = self.path / "run.tmp"
        temporary.write_text(self.model_dump_json(indent=2) + "\n")
        temporary.replace(self.path / "run.json")

    @classmethod
    def load(cls, run_name: str) -> typing.Self:
        """Load a named run."""
        path = cli.RunConfig(run_name=run_name).run_dir
        run = cls.model_validate_json((path / "run.json").read_text())

        if run.run_name != run_name:
            raise ValueError("Run name does not match its folder")

        return run

    def log(self, episode: Episode) -> None:
        """Append one completed episode."""
        with (self.path / "metrics.csv").open("a", newline="") as file:
            csv.DictWriter(
                file,
                fieldnames=list(Episode.model_fields),
            ).writerow(episode.model_dump())

    def save_weights(self, policy: agent.PolicyAgent) -> None:
        """Save the final policy and value weights."""
        if (policy.observation_size, policy.n_actions) != (self.observation_size, self.n_actions):
            raise ValueError("Agent dimensions do not match the run")

        weights = Weights(
            policy=policy.policy_net.state_dict(),
            value=policy.value_net.state_dict(),
        )
        temporary = self.path / "weights.tmp"
        torch.save(weights.model_dump(), temporary)
        temporary.replace(self.path / "weights.pt")

    def load_weights(self, *, device: str = "cpu") -> agent.PolicyAgent:
        """Load an agent with this run's final weights."""
        weights = Weights.model_validate(
            torch.load(
                self.path / "weights.pt",
                map_location=device,
                weights_only=True,
            )
        )

        policy = agent.PolicyAgent(
            self.observation_size,
            self.n_actions,
            lr_value=self.settings.lr_value,
            lr_policy=self.settings.lr_policy,
            gamma=self.settings.gamma,
            beta=self.settings.beta,
            device=device,
            adam_eps=self.settings.adam_eps,
        )
        policy.policy_net.load_state_dict(weights.policy)
        policy.value_net.load_state_dict(weights.value)
        return policy
