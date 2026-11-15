import contextlib
import csv
import json
import pathlib
import tempfile
import typing
import unittest
import unittest.mock

import gymnasium as gym
import numpy as np
import pydantic
import pydantic_settings
import torch

from a2c import agent, model, transitions
from experiments import analyse, config, envs, runs, scores, train, utils


class EndingEnv(gym.Env):
    """The final state differs from the reset state, exposing premature resets."""

    def __init__(self, terminated):
        self.terminated = terminated

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        return np.array([7.0], dtype=np.float32), 1.0, self.terminated, not self.terminated, {}


class FixedAgent(agent.PolicyAgent):
    def actions(self, states, *, stochastic=True):
        return np.zeros(np.asarray(states).shape[0], dtype=np.int64)


class CoreTests(unittest.TestCase):
    def test_orthogonal_linear_initialization(self):
        for inputs, outputs in [(5, 3), (3, 5)]:
            with self.subTest(inputs=inputs, outputs=outputs):
                layer = model.OrthogonalLinear(inputs, outputs, gain=0.5)
                weights = layer.weight.detach()
                gram = weights @ weights.T if outputs <= inputs else weights.T @ weights
                torch.testing.assert_close(
                    gram, 0.25 * torch.eye(min(inputs, outputs)), atol=1e-6, rtol=1e-5
                )
                torch.testing.assert_close(layer.bias, torch.zeros(outputs))

    def test_final_observation_survives_reset(self):
        generator = iter(
            transitions.GenerateTransitions(
                [EndingEnv(True), EndingEnv(False)], FixedAgent(1, 2), 42
            )
        )
        first = next(generator)
        np.testing.assert_equal(first.next_states, [[7.0], [7.0]])
        np.testing.assert_equal(first.terminated, [True, False])
        np.testing.assert_equal(first.truncated, [False, True])
        np.testing.assert_equal(next(generator).states, [[0.0], [0.0]])

    def test_only_termination_stops_bootstrapping(self):
        policy = agent.PolicyAgent(1, 2, gamma=0.5)
        with torch.no_grad():
            for parameter in policy.value_net.parameters():
                parameter.zero_()
            output = policy.value_net.ffl[-1]
            assert isinstance(output, torch.nn.Linear)
            assert output.bias is not None
            output.bias.fill_(4.0)
        targets = policy.compute_targets([[7.0], [7.0]], [1.0, 1.0], [True, False])
        torch.testing.assert_close(targets, torch.tensor([1.0, 3.0]))

    def test_single_environment_update_stays_finite(self):
        policy = agent.PolicyAgent(1, 2)
        transition = next(iter(transitions.GenerateTransitions([EndingEnv(False)], policy, 42)))
        losses = policy.update(transition)
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))

    def test_invalid_config_and_path_names_are_rejected(self):
        for values in [
            {"seed": -1},
            {"lr_value": 0},
            {"lr_policy": -1},
            {"gamma": 1.1},
            {"beta": -0.1},
            {"adam_eps": 0},
            {"device": "invalid"},
            {"total_steps": 0},
            {"max_episode_steps": 0},
            {"max_episode_steps": -1},
            {"env": ""},
            {"env": "Pendulum-v1"},
            {"env": "LunarLander-v2"},
            {"run_name": "../weights"},
            {"run_name": "/tmp/run"},
        ]:
            with self.subTest(values=values), self.assertRaises(pydantic.ValidationError):
                config.TrainConfig.model_validate({"run_name": "test"} | values)

    def test_supported_environments_have_expected_spaces(self):
        for name in typing.get_args(config.Environment.__value__):
            with (
                self.subTest(env=name),
                envs.open(config.TrainConfig(run_name="spaces", env=name)) as env,
            ):
                assert isinstance(env.observation_space, gym.spaces.Box)
                self.assertEqual(len(env.observation_space.shape), 1)
                assert isinstance(env.action_space, gym.spaces.Discrete)
                self.assertEqual(env.action_space.start, 0)

    def test_discrete_observation_adapter_preserves_states_and_rewards(self):
        names: list[config.Environment] = ["FrozenLake-v1", "Taxi-v4"]
        for name in names:
            cfg = config.TrainConfig(run_name="one-hot", env=name)
            with (
                self.subTest(env=name),
                contextlib.closing(gym.make(name)) as original,
                envs.open(cfg) as wrapped,
            ):
                assert isinstance(original.observation_space, gym.spaces.Discrete)
                state, _ = original.reset(seed=42)
                observation, _ = wrapped.reset(seed=42)
                self.assertEqual(observation.dtype, np.float32)
                self.assertEqual(observation.shape, (original.observation_space.n,))
                np.testing.assert_equal(
                    observation, np.eye(original.observation_space.n, dtype=np.float32)[state]
                )
                state, reward, terminated, truncated, _ = original.step(0)
                observation, actual_reward, actual_terminated, actual_truncated, _ = wrapped.step(0)
                np.testing.assert_equal(
                    observation, np.eye(original.observation_space.n, dtype=np.float32)[state]
                )
                self.assertEqual(
                    (actual_reward, actual_terminated, actual_truncated),
                    (reward, terminated, truncated),
                )

    def test_discrete_runs_train_save_and_evaluate(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(config, "RUNS_DIR", pathlib.Path(directory)),
            unittest.mock.patch.object(train, "N_ENVS", 2),
        ):
            cases: list[tuple[config.Environment, int, int]] = [
                ("FrozenLake-v1", 16, 100),
                ("Taxi-v4", 500, 200),
            ]
            for name, size, limit in cases:
                with self.subTest(env=name):
                    cfg = config.TrainConfig(
                        run_name=name.lower(), env=name, total_steps=4, device="cpu"
                    )
                    run = train.train(cfg)
                    loaded = runs.Run.load(run.run_name)
                    self.assertEqual(loaded.observation_size, size)
                    report = analyse.evaluate(loaded.load_weights(), loaded, 1)
                    self.assertEqual(report.env, name)
                    self.assertLessEqual(report.lengths[0], limit)

    def test_box_training_keeps_normal_starts(self):
        cfg = config.TrainConfig(run_name="box")
        with envs.open_many(cfg, count=1) as training, envs.open(cfg) as evaluation:
            for seed in range(5):
                actual, _ = training[0].reset(seed=seed)
                expected, _ = evaluation.reset(seed=seed)
                np.testing.assert_equal(actual, expected)

    def test_exploring_starts_are_safe_and_only_used_for_training(self):
        cfg = config.TrainConfig(run_name="starts", env="CliffWalking-v1")
        with envs.open_many(cfg, count=1) as training, envs.open(cfg) as evaluation:
            starts = {int(training[0].reset(seed=seed)[0].argmax()) for seed in range(100)}
            self.assertGreater(len(starts), 1)
            self.assertTrue(starts.isdisjoint(range(37, 48)))  # cliffs and terminal goal
            for seed in range(5):
                observation, _ = evaluation.reset(seed=seed)
                self.assertEqual(observation.argmax(), 36)  # standard bottom-left start
            assert evaluation.spec is not None
            self.assertEqual(evaluation.spec.max_episode_steps, 200)
            terminated, truncated = False, False
            for _ in range(200):
                _, _, terminated, truncated, _ = evaluation.step(3)  # stay at the left wall
            self.assertFalse(terminated)
            self.assertTrue(truncated)

    def test_optimizer_settings_round_trip(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(config, "RUNS_DIR", pathlib.Path(directory)),
        ):
            cfg = config.TrainConfig(run_name="optimizer", adam_eps=1e-5)
            policy = agent.PolicyAgent(2, 3, adam_eps=cfg.adam_eps)
            run = runs.Run.create(cfg, policy, 64)
            run.save_weights(policy)
            loaded = runs.Run.load(run.run_name).load_weights()
            torch.testing.assert_close(
                policy.policy_net.state_dict(), loaded.policy_net.state_dict()
            )
            torch.testing.assert_close(policy.value_net.state_dict(), loaded.value_net.state_dict())
            self.assertEqual(loaded.optimizer_policy.param_groups[0]["eps"], 1e-5)

    def test_scaling_handles_all_supported_spaces(self):
        for name in typing.get_args(config.Environment.__value__):
            with (
                self.subTest(env=name),
                envs.open(config.TrainConfig(run_name="raw", env=name)) as original,
                envs.open(
                    config.TrainConfig(run_name="scaled", env=name, scale_observations=True)
                ) as scaled,
            ):
                raw, _ = original.reset(seed=42)
                actual, _ = scaled.reset(seed=42)
                space = original.observation_space
                assert isinstance(space, gym.spaces.Box)
                bounded = np.isfinite(space.low) & np.isfinite(space.high)
                expected = raw.copy()
                expected[bounded] = (
                    2
                    * (raw[bounded] - space.low[bounded])
                    / (space.high[bounded] - space.low[bounded])
                    - 1
                )
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                np.testing.assert_equal(actual[~bounded], raw[~bounded])
                self.assertTrue(np.all(np.isfinite(actual)))
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(scaled.observation_space.contains(actual))
                assert scaled.spec is not None and original.spec is not None
                self.assertEqual(scaled.spec.max_episode_steps, original.spec.max_episode_steps)

    def test_environment_closes_when_wrapper_setup_fails(self):
        env = gym.make("CartPole-v1")
        cfg = config.TrainConfig(run_name="cleanup", scale_observations=True)
        with (
            unittest.mock.patch.object(gym, "make", return_value=env),
            unittest.mock.patch.object(env, "close", wraps=env.close) as close,
            unittest.mock.patch.object(
                gym.wrappers, "RescaleObservation", side_effect=RuntimeError("wrapper failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "wrapper failed"):
                with envs.open(cfg):
                    self.fail("Wrapper creation should fail before yielding")
            close.assert_called_once()

    def test_environment_batch_closes_after_partial_creation_or_body_failure(self):
        cfg = config.TrainConfig(run_name="cleanup")
        for creation_fails in [True, False]:
            first, second = gym.make(cfg.env), gym.make(cfg.env)
            with (
                self.subTest(creation_fails=creation_fails),
                unittest.mock.patch.object(first, "close", wraps=first.close) as close_first,
                unittest.mock.patch.object(second, "close", wraps=second.close) as close_second,
                unittest.mock.patch.object(
                    gym, "make", side_effect=[first, second, RuntimeError("creation failed")]
                ),
            ):
                with self.assertRaises(RuntimeError):
                    with envs.open_many(cfg, count=3 if creation_fails else 2):
                        raise RuntimeError("body failed")
                close_first.assert_called_once()
                close_second.assert_called_once()

    def test_analysis_restores_scaling_but_not_training_limit(self):
        class CapturingAgent(agent.PolicyAgent):
            def __init__(self):
                super().__init__(2, 3)
                self.states = []

            def actions(self, states, *, stochastic=False):
                self.states.append(np.asarray(states))
                return np.zeros(np.asarray(states).shape[0], dtype=np.int64)

        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(config, "RUNS_DIR", pathlib.Path(directory)),
            unittest.mock.patch.object(train, "N_ENVS", 1),
        ):
            cfg = config.TrainConfig(
                run_name="scaled",
                env="MountainCar-v0",
                total_steps=6,
                scale_observations=True,
                max_episode_steps=3,
                device="cpu",
            )
            run = train.train(cfg)
            with (run.path / "metrics.csv").open() as file:
                self.assertEqual(
                    [int(row["episode_length"]) for row in csv.DictReader(file)], [3, 3]
                )
            run = runs.Run.load(run.run_name)
            self.assertTrue(run.settings.scale_observations)
            self.assertEqual(run.settings.max_episode_steps, 3)
            policy = CapturingAgent()
            report = analyse.evaluate(policy, run, 1)
            self.assertEqual(report.lengths, [200])
            with envs.open(cfg) as env:
                expected, _ = env.reset(seed=scores.EVALUATION_SEED)
            np.testing.assert_equal(policy.states[0], [expected])

    def test_run_name_cannot_overwrite_existing_run(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(config, "RUNS_DIR", pathlib.Path(directory)),
        ):
            cfg = config.TrainConfig(run_name="test-run")
            policy = agent.PolicyAgent(4, 2)
            run = runs.Run.create(cfg, policy, n_envs=1)
            previous = (run.path / "run.json").read_bytes()
            with self.assertRaises(FileExistsError):
                runs.Run.create(cfg, policy, n_envs=1)
            self.assertEqual((run.path / "run.json").read_bytes(), previous)

    def test_analysis_uses_run_name_and_saved_environment(self):
        make_env = gym.make

        def make_short_env(name: str, **kwargs: typing.Any) -> gym.Env:
            kwargs["max_episode_steps"] = 3
            return make_env(name, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(config, "RUNS_DIR", pathlib.Path(directory)),
            unittest.mock.patch.object(
                gym,
                "make",
                side_effect=make_short_env,
            ),
        ):
            cfg = config.TrainConfig(run_name="saved-run", env="MountainCar-v0")
            policy = agent.PolicyAgent(2, 3)
            run = runs.Run.create(cfg, policy, n_envs=1)
            run.save_weights(policy)
            options = pydantic_settings.CliApp.run(
                config.AnalyseConfig, cli_args=["--run-name", "saved-run", "--episodes", "2"]
            )
            report = analyse.analyse(options)
            self.assertEqual(report.env, "MountainCar-v0")
            self.assertEqual(report.lengths, [3, 3])
            for name in ("evaluation.json", "training.png"):
                self.assertTrue((cfg.run_dir / "analysis" / name).exists())

    def test_record_and_watch_share_one_episode(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(config, "RUNS_DIR", pathlib.Path(directory)),
        ):
            policy = agent.PolicyAgent(4, 2)
            run = runs.Run.create(
                config.TrainConfig(run_name="demo", scale_observations=True, max_episode_steps=1),
                policy,
                n_envs=1,
            )
            for record, watch in [(True, False), (False, True), (True, True)]:
                output = run.path / f"demo-{record}-{watch}"
                output.mkdir()
                env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=3)
                with (
                    self.subTest(record=record, watch=watch),
                    unittest.mock.patch.object(gym, "make", return_value=env) as make,
                    unittest.mock.patch.object(env, "reset", wraps=env.reset) as reset,
                    unittest.mock.patch.object(env, "step", wraps=env.step) as step,
                    unittest.mock.patch.object(
                        analyse, "episode", wraps=analyse.episode
                    ) as rollout,
                    unittest.mock.patch.object(
                        gym.wrappers.HumanRendering, "_render_frame", autospec=True
                    ) as display,
                ):
                    analyse.demo_episode(policy, run, output, record=record, watch=watch)
                    make.assert_called_once()
                    reset.assert_called_once()
                    self.assertEqual(reset.call_args.kwargs["seed"], scores.EVALUATION_SEED)
                    self.assertEqual(step.call_count, 3)
                    rollout.assert_called_once()
                    self.assertEqual(display.call_count, 4 if watch else 0)
                    wrapper = rollout.call_args.args[1]
                    if watch:
                        self.assertIsInstance(wrapper, gym.wrappers.HumanRendering)
                        wrapper = wrapper.env
                    if record:
                        self.assertIsInstance(wrapper, gym.wrappers.RecordVideo)
                        self.assertGreater((output / "episode.mp4").stat().st_size, 0)
                    else:
                        self.assertFalse((output / "episode.mp4").exists())
                    self.assertIsInstance(
                        wrapper.env if record else wrapper, gym.wrappers.RescaleObservation
                    )

    def test_training_saves_overridden_hyperparameters(self):
        make_env = gym.make

        def make_short_env(name: str, **kwargs: typing.Any) -> gym.Env:
            kwargs["max_episode_steps"] = 1
            return make_env(name, **kwargs)

        save_weights = runs.Run.save_weights
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(config, "RUNS_DIR", pathlib.Path(directory)),
            unittest.mock.patch.object(train, "N_ENVS", 1),
            unittest.mock.patch.object(
                runs.Run, "save_weights", autospec=True, side_effect=save_weights
            ) as save,
            unittest.mock.patch.object(
                gym,
                "make",
                side_effect=make_short_env,
            ),
        ):
            cfg = pydantic_settings.CliApp.run(
                config.TrainConfig,
                cli_args=[
                    "--run-name",
                    "configured",
                    "--total-steps",
                    "3",
                    "--device",
                    "cpu",
                    "--lr-value",
                    "0.0002",
                    "--lr-policy",
                    "0.0003",
                    "--gamma",
                    "0.95",
                    "--beta",
                    "0.02",
                ],
            )
            run = train.train(cfg)
            save.assert_called_once()
            saved = runs.Run.load(cfg.run_name)
            policy = saved.load_weights()
            self.assertEqual(saved.observation_size, 4)
            for name in ("lr_value", "lr_policy", "gamma", "beta"):
                self.assertEqual(getattr(saved.settings, name), getattr(cfg, name))
            self.assertEqual(policy.optimizer_value.param_groups[0]["lr"], cfg.lr_value)
            self.assertEqual(policy.optimizer_policy.param_groups[0]["lr"], cfg.lr_policy)
            manifest = json.loads((run.path / "run.json").read_text())
            self.assertNotIn("device", manifest["settings"])
            self.assertNotIn("device", manifest)
            self.assertNotIn("packages", manifest)

    def test_final_weights_round_trip(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(config, "RUNS_DIR", pathlib.Path(directory)),
        ):
            cfg = config.TrainConfig(run_name="final")
            policy = agent.PolicyAgent(4, 2)
            run = runs.Run.create(cfg, policy, n_envs=1)
            run.save_weights(policy)
            loaded = runs.Run.load("final").load_weights()
            torch.testing.assert_close(
                loaded.policy_net.state_dict(), policy.policy_net.state_dict()
            )
            torch.testing.assert_close(loaded.value_net.state_dict(), policy.value_net.state_dict())
            self.assertEqual(list(run.path.rglob("*.pt")), [run.path / "weights.pt"])

    def test_scorecard_uses_only_selected_runs_and_explicit_targets(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(config, "RUNS_DIR", pathlib.Path(directory) / "runs"),
        ):
            cases: list[tuple[str, config.Environment, list[float]]] = [
                ("cart", "CartPole-v1", [500.0, 500.0]),
                ("acro", "Acrobot-v1", [-90.0, -80.0]),
                ("cliff", "CliffWalking-v1", [-13.0, -13.0]),
                ("unselected", "MountainCar-v0", [-200.0, -200.0]),
            ]
            for name, env, returns in cases:
                run = runs.Run.create(
                    config.TrainConfig(run_name=name, env=env), agent.PolicyAgent(4, 2), 64
                )
                analysis = run.path / "analysis"
                analysis.mkdir()
                report = scores.Evaluation(
                    run_name=name, env=env, seeds=[1000, 1001], returns=returns, lengths=[1, 1]
                )
                (analysis / "evaluation.json").write_text(report.model_dump_json())
            output = pathlib.Path(directory) / "published"
            path = scores.write_scores(["cart", "acro", "cliff"], output)
            text = path.read_text()
            self.assertIn("| CartPole-v1 | 475 | 500.00 ± 0.00 | ✓ |", text)
            self.assertIn("| Acrobot-v1 | -100 | -85.00 ± 5.00 | ✓ |", text)
            self.assertIn("| CliffWalking-v1 | -13 | -13.00 ± 0.00 | ✓ |", text)
            self.assertIn("no official threshold", text)
            self.assertNotIn("MountainCar", text)
            self.assertEqual({path.name for path in output.iterdir()}, {"scores.md"})
            previous = path.read_bytes()
            evaluation_path = config.RUNS_DIR / "cart" / "analysis" / "evaluation.json"
            report = scores.Evaluation.model_validate_json(evaluation_path.read_text())
            bad = report.model_copy(update={"run_name": "wrong"})
            evaluation_path.write_text(bad.model_dump_json())
            with self.assertRaises(ValueError):
                scores.write_scores(["cart"], output)
            self.assertEqual(path.read_bytes(), previous)

    def test_evaluation_rejects_incomplete_or_invalid_results(self):
        valid = dict(
            run_name="test",
            env="CartPole-v1",
            seeds=[1000, 1001],
            returns=[1.0, 2.0],
            lengths=[1, 2],
        )
        for changes in [
            {"seeds": [], "returns": [], "lengths": []},
            {"lengths": [1]},
            {"returns": [float("nan"), 1.0]},
            {"returns": [float("inf"), 1.0]},
            {"seeds": [1000, 1002]},
        ]:
            with self.subTest(changes=changes), self.assertRaises(pydantic.ValidationError):
                scores.Evaluation(**(valid | changes))

    def test_device_resolution(self):
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(utils.choose_device("auto"), "cpu")
            self.assertEqual(utils.choose_device("cpu"), "cpu")
            with self.assertRaises(ValueError):
                utils.choose_device("cuda")
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(utils.choose_device("auto"), "cuda")
        with unittest.mock.patch.object(torch.backends.mps, "is_available", return_value=False):
            with self.assertRaises(ValueError):
                utils.choose_device("mps")


if __name__ == "__main__":
    unittest.main()
