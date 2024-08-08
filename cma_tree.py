import cma
import numpy as np


DEPTH = 3


def fitness(x, X, y):
    dt = x.reshape((2**DEPTH)-1, -1)
    indices = np.zeros(len(X), dtype=int)

    bias_X = np.concatenate((X, np.ones(len(X)).reshape(-1, 1)), axis=1)

    for d in range(DEPTH):
        outcome = np.einsum('ab,ab->a', bias_X, dt[indices])
        indices = 2 * indices + np.where(outcome < 0, 1, 2)

    errors = 0
    for i in np.unique(indices):
        idx = (indices == i)

        if len(idx) > 0:
            ys = y[idx]

            best = None
            max_ = -1
            for j in np.unique(ys):
                sum_ = sum(ys == j)

                if sum_ > max_:
                    best = j
                    max_ = sum_


            errors += sum(ys != best)
    return errors / len(X)


def inference(dt, X):
    indices = np.zeros(len(X), dtype=int)
    bias_X = np.concatenate((X, np.ones(len(X)).reshape(-1, 1)), axis=1)

    for d in range(DEPTH):
        outcome = np.einsum('ab,ab->a', bias_X, dt[indices])
        indices = 2 * indices + np.where(outcome < 0, 1, 2)

    return indices


def train(dt, X, y):
    classes = -np.ones(len(X), dtype=int)
    errors = 0
    indices = inference(dt, X)

    for i in np.unique(indices):
        idx = (indices == i)

        if len(idx) > 0:
            ys = y[idx]

            best = None
            max_ = -1
            for j in np.unique(ys):
                sum_ = sum(ys == j)

                if sum_ > max_:
                    best = j
                    max_ = sum_

            classes[i] = best
            errors += sum(ys != best)

    return classes, errors / len(X)


if __name__ == "__main__":
    import argparse
    import torch
    from functools import partial
    import gymnasium as gym
    from sac import setup_sac
    from utils import init_cfg, load_agent

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker")
    parser.add_argument("--version", type=str, default="seed0")
    args = parser.parse_args()

    cfg = init_cfg(f"runs/{args.env}/{args.version}/config.yml")
    envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(cfg.env_id))])
    agent, _ = load_agent(setup_sac(cfg, envs), f"runs/{args.env}/{args.version}/models/checkpoint_final")

    X = np.load(f"runs/{args.env}/{args.version}/models/checkpoint_final/observations.npz")["array"].squeeze()

    y = []
    for batch in np.split(X, 200):
            expert_idx = agent.actor.router(torch.tensor(batch, dtype=torch.float32, device=cfg.device)).cpu().detach().numpy()
            y.append(np.argmax(expert_idx, -1))

    X = X.squeeze()[:100]
    y = np.concatenate(y)[:100]

    mean, std = X.mean(0), X.std(0)
    X = (X - mean)/(std)

    n_params = (X.shape[-1] + 1) * (2 ** DEPTH - 1)

    part_f = partial(fitness, X=X, y=y)

    es = cma.CMAEvolutionStrategy([0] * n_params, 1e0, {'popsize': 64})

    res = es.optimize(part_f, n_jobs=10)
    xbest = res.result.xbest
    dt = xbest.reshape((2**DEPTH)-1, -1)

    classes, error = train(dt, X, y)
    print(error)

    np.savez_compressed(f"{args.env}_{args.version}_dt.npz", array=dt)
    np.savez_compressed(f"{args.env}_{args.version}_classes.npz", array=classes)
    np.savez_compressed(f"{args.env}_{args.version}_classes.npz", array=classes)
    np.savez_compressed(f"{args.env}_{args.version}_mean_std.npz", array=np.array([mean, std]))

    obs, _ = envs.reset()

    rews = []
    for _ in range(100):
        while True:
            expert_idx = inference(dt, (obs - mean) / std)
            actions = agent.actor.mean_experts[classes[int(expert_idx)]](torch.tensor(obs, dtype=torch.float32, device=cfg.device))
            actions = actions.cpu().detach().numpy()

            next_obs, _, _, _, infos = envs.step(actions)

            if "final_info" in infos:
                rews.append(infos["final_info"][0]['episode']['r'][0])
                break

            obs = next_obs

    print(np.mean(rews))
