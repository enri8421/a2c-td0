import pathlib
import typing

import pydantic
import pydantic_settings

from experiments import utils

type Environment = typing.Literal[
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "LunarLander-v3",
    "FrozenLake-v1",
    "Taxi-v4",
    "CliffWalking-v1",
]

RUNS_DIR = pathlib.Path(__file__).resolve().parents[1] / "runs"


class RunConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_enforce_required=True,
        env_prefix="A2C_",
    )

    device: utils.DeviceType = "auto"
    run_name: str = pydantic.Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")

    @property
    def run_dir(self) -> pathlib.Path:
        return RUNS_DIR / self.run_name


class TrainConfig(RunConfig):
    env: Environment = "CartPole-v1"
    seed: int = pydantic.Field(default=42, ge=0, le=2**32 - 1)

    total_steps: int = pydantic.Field(default=500_000, ge=1)
    scale_observations: bool = False
    adam_eps: float = pydantic.Field(default=1e-3, gt=0, allow_inf_nan=False)
    max_episode_steps: int | None = pydantic.Field(default=None, ge=1)

    lr_value: float = pydantic.Field(default=0.001, gt=0, allow_inf_nan=False)
    lr_policy: float = pydantic.Field(default=0.001, gt=0, allow_inf_nan=False)
    gamma: float = pydantic.Field(default=0.99, ge=0, le=1)
    beta: float = pydantic.Field(default=0.01, ge=0, allow_inf_nan=False)


class AnalyseConfig(RunConfig):
    episodes: int = pydantic.Field(default=100, ge=1)
    record: bool = False
    watch: bool = False
