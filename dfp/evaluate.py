import time
import tqdm
import torch
import pandas as pd
from sacred import Ingredient

from dfp.utils import save_model


evaluator = Ingredient('evaluator')


@evaluator.capture
def evaluate_policy(env, model, policy, n_eval_episodes, epoch, run_dir, _log, _run):

    logger = _log

    # EVALUATION
    logger.info(f"Evaluating ...")
    eval_tic = time.time()
    model.eval()

    eval_metrics = []
    with torch.no_grad():

        for _ in tqdm.trange(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            while not done:
                action = policy(obs)
                next_obs, reward, done, info = env.step(action)

                obs = next_obs
                episode_steps += 1
                episode_reward += reward

            # end of episode
            info['ep_return'] = episode_reward
            info['ep_length'] = episode_steps
            eval_metrics.append(info)

    eval_toc = time.time()
    logger.info(f"Evaluated in {eval_toc - eval_tic:.2f}s.")

    # save detailed epoch test metrics
    test_metrics_df = pd.DataFrame.from_records(eval_metrics)
    test_metrics_path = run_dir / f"test_metrics_epoch_{epoch:03}.csv"
    test_metrics_df.round(4).to_csv(test_metrics_path, index=False)
    _run.add_artifact(test_metrics_path)

    # return average metrics
    mean_test_metrics = test_metrics_df.mean().round(4).to_dict()

    # save model
    model_path = run_dir / f"model_epoch_{epoch:03}.pth"
    logger.info(f"Saving model to {model_path}")
    info = {'epoch': epoch}
    save_model(model, model_path, **info)
    _run.add_artifact(model_path)

    return mean_test_metrics
