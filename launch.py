import torch
import logging
import tempfile
import numpy as np
from sacred import Experiment
from pathlib import Path
from tabulate import tabulate

from dfp.doom_sim_env import DoomSimulatorEnv
from dfp.env_wrapper import TargetMeasEnvWrapper
from dfp.network import model, make_model
from dfp.agent import Agent
from dfp.replay_buffer import DFPReplay
from dfp.policies import EpsilonGreedyPolicy, DFPPolicy, LinearSchedule
from dfp.preprocessing import ObservationPreprocessor, BatchObservationPreprocessor
from dfp.evaluate import evaluator, evaluate_policy
from dfp.utils import get_logger


MAPS_DIR = Path(__file__).parent / 'maps'

ex = Experiment(name='DFP', ingredients=[model, evaluator])
ex.logger = get_logger(__name__, level=logging.INFO)


@ex.config
def cfg():
    # env
    scenario = 'D1_basic'
    image_shape = [84, 84]
    frame_skip = 4
    maps = ['MAP01']
    switch_maps = False

    # training loop
    n_train_steps = 800_000
    n_eval_episodes = 100
    train_freq = 64    # env steps
    test_freq = 7812   # grad steps
    log_freq = 100     # grad steps
    init_eval = True

    # DFP / Q-learning
    replay_capacity = 20_000
    min_horizon = 4
    epsilon_start = 1.0
    epsilon_end = 0.15
    exploration_frac = 1.0
    future_steps = [1, 2, 4, 8, 16, 32]
    temporal_coeffs = [0, 0, 0, 0.5, 0.5, 1.0]
    meas_coeffs = [1.]
    target_meas_scale = [30.]
    sample_goals = False
    goal_space = 'pos_neg'  # pos_neg | pos

    # optimization
    batch_size = 64
    lr = 0.0001
    scheduler_step = 250_000
    scheduler_decay = 0.3

    # other
    device = 'cpu'
    seed = 1


@ex.capture
def report_metrics(step, metrics, global_vars, _run, _log, mode='test'):

    # prefix with mode
    prefix = mode + '_'
    metrics_report = {prefix + key: value for key, value in metrics.items()}
    metrics_report.update(global_vars)

    # REPORT EVALUATION METRICS
    _log.info(f"{mode.capitalize()} report:")
    _log.info(f"\n{tabulate(metrics_report.items(), tablefmt='grid')}")
    _log.info("-" * 40)

    for key, value in metrics_report.items():
        _run.log_scalar(key, round(value, 4), step)


@ex.automain
def train(scenario, image_shape, frame_skip, maps, switch_maps, n_train_steps, n_eval_episodes,
          train_freq, test_freq, log_freq, init_eval, replay_capacity, min_horizon, epsilon_start, epsilon_end,
          exploration_frac, future_steps, temporal_coeffs, meas_coeffs, target_meas_scale, sample_goals, goal_space,
          batch_size, lr, scheduler_step, scheduler_decay, device, seed, _log, _run):

    if device.startswith('cuda'):
        assert torch.cuda.is_available()

    doom_config_path = MAPS_DIR / (scenario + '.cfg')
    assert doom_config_path.exists()
    doom_config_path = str(doom_config_path)

    logger = _log
    device = torch.device(device)

    # objective
    assert len(temporal_coeffs) == len(future_steps)
    temporal_coeffs = np.asarray(temporal_coeffs, dtype=np.float32)
    future_steps = np.asarray(future_steps, dtype=np.int64)
    meas_coeffs = np.asarray(meas_coeffs, dtype=np.float32)
    target_scale = np.tile(target_meas_scale, len(future_steps)).astype(np.float32)

    # environment
    def _make_env(sample_goals_, rank=0):
        sim = DoomSimulatorEnv(doom_config_path, frame_skip=frame_skip, switch_maps=switch_maps, maps=maps)
        env = TargetMeasEnvWrapper(sim, meas_coeffs, temporal_coeffs, goal_space=goal_space, sample_goals=sample_goals_)
        env.seed(seed + rank)
        return env

    train_env = _make_env(sample_goals)
    test_env = _make_env(False)

    # model
    net_image_shape = [1, *image_shape]  # env.observation_space
    dim_meas = len(meas_coeffs)
    dim_goal = len(meas_coeffs) * len(temporal_coeffs)
    model = make_model(image_shape=net_image_shape, dim_goal=dim_goal, dim_meas=dim_meas,
                       num_actions=train_env.action_space.n).to(device)

    # policy
    obs_transform = ObservationPreprocessor(image_shape=image_shape, meas_scale=100, device=device)
    greedy_policy = DFPPolicy(model, obs_fn=obs_transform)
    exploration_steps = int(exploration_frac * n_train_steps)
    exploration = LinearSchedule(epsilon_start, epsilon_end, exploration_steps)
    random_policy = lambda x: train_env.action_space.sample()
    collect_policy = EpsilonGreedyPolicy(greedy_policy, random_policy, exploration)

    # training scheme
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_decay)

    # replay buffer
    target_transform = lambda x: torch.from_numpy(x / target_scale).to(device)
    batch_obs_transform = BatchObservationPreprocessor(image_shape=image_shape, meas_scale=100, device=device)
    replay_buffer = DFPReplay(replay_capacity, ob_space=train_env.observation_space, future_steps=future_steps,
                              min_horizon=min_horizon, obs_fn=batch_obs_transform, target_fn=target_transform,
                              device=device)

    # agent
    agent = Agent(train_env, collect_policy, replay_buffer, model, optimizer, scheduler)

    # fill up replay buffer
    logger.info(f"Filling up the replay buffer...")
    agent.fill_buffer()
    logger.info(f"Replay buffer is full: [{len(replay_buffer)} / {replay_capacity}]")

    perf_metric = 'ep_return'
    best_perf = 0

    # make a temp dir to store models, metrics, etc.
    with tempfile.TemporaryDirectory() as temp_dir:

        run_dir = Path(temp_dir)

        # eval once before training
        if init_eval:
            test_metrics = evaluate_policy(test_env, model, greedy_policy, n_eval_episodes, epoch=0, run_dir=run_dir)
            best_perf = test_metrics[perf_metric]
            global_vars = dict(epsilon=epsilon_start, **agent.counters)
            report_metrics(0, test_metrics, global_vars, mode='test')

        # loop
        epoch = 1
        for t in range(1, n_train_steps + 1):

            # TRAINING STEP
            agent.train_step(batch_size)
            epsilon = exploration.step()

            # COLLECT DATA
            for _ in range(train_freq):
                agent.env_step()

            # REPORT TRAINING METRICS
            if t % log_freq == 0:
                train_metrics = agent.gather_metrics()
                global_vars = dict(epsilon=epsilon, **agent.counters)
                report_metrics(t, train_metrics, global_vars, mode='train')

            # EVALUATION
            if t % test_freq == 0:
                test_metrics = evaluate_policy(test_env, model, greedy_policy, n_eval_episodes, epoch=epoch,
                                              run_dir=run_dir)
                epoch_perf = test_metrics[perf_metric]
                global_vars = dict(epsilon=epsilon, **agent.counters)
                report_metrics(t, test_metrics, global_vars, mode='test')

                # epoch performance
                if epoch_perf > best_perf:
                    logger.info(f"{perf_metric} improved: {best_perf:.4f} --> {epoch_perf:.4f}\n")
                    best_perf = epoch_perf

                logger.info(f"Finished epoch {epoch:3} (best performance: {best_perf:.4f})")
                logger.info("-" * 80)
                epoch += 1

    train_env.close()
    test_env.close()

    return best_perf
