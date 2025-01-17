import time
import torch
import wandb
import gymnasium as gym
import logging
import os

from sac import train_sac, setup_sac
from utils import init_cfg, save_agent, load_agent


def log_episode_info(cfg, global_step, info, agent):
    episode_length = info["episode"]["l"][0]
    log_dict = {
        "global_step": global_step,
        "episodic_return": info["episode"]["r"][0],
        "episodic_length": episode_length,
        **{f"episodic_expert_{i}_ratio": agent.actor.episodic_expert_count[i] / episode_length for i in range(cfg.sac.n_experts)},
    }
    if cfg.log.wandb:
        wandb.log(log_dict)
    if cfg.log.log_local:
        logging.info(",".join(f"{value}" for _, value in log_dict.items()))


def train(cfg):
    env = gym.make(cfg.env_id)
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)
    envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(env)])

    agent = setup_sac(cfg, envs)

    # load checkpoint or initialize replay buffer with random actions
    if cfg.checkpoint:
        agent, _ = load_agent(agent, cfg.checkpoint)
        print(f"Model-checkpoint loaded")
    else:
        obs, _ = envs.reset(seed=cfg.seed)
        for _ in range(cfg.learning_starts):
            actions = envs.action_space.sample()
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            agent.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            obs = next_obs
        print(f"RB initialized with {cfg.learning_starts} samples")

    obs, _ = envs.reset(seed=cfg.seed+1)

    for global_step in range(cfg.learning_starts, cfg.total_timesteps):
        actions = agent.actor.get_action(torch.tensor(obs, dtype=torch.float32, device=cfg.device))
        actions = actions[0].cpu().detach().numpy()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            info = infos["final_info"][0]
            print(f"STEP : {global_step}/{cfg.total_timesteps}".ljust(25), f"REWARD : {info['episode']['r'][0]:.3f}".ljust(20), f"LENGTH : {info['episode']['l'][0]}".ljust(20))
            log_episode_info(cfg, global_step, info, agent)
            agent.actor.episodic_expert_count.fill_(0)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        agent.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        train_sac(cfg, agent)

        if cfg.log.save_models and (global_step + 1) % int(5e4) == 0:
            training_finished = global_step + 1 == cfg.total_timesteps
            checkpoint = f"checkpoint_{'final' if training_finished else f'{(global_step+1)//1000}k'}"
            save_agent(
                agent,
                f"{cfg.run_path}/models/{checkpoint}",
                save_obs=True if training_finished else False,
            )
            print(f"{checkpoint} saved")

    envs.close()
    return 0


if __name__ == "__main__":
    import numpy as np
    import argparse
    import random

    # ---- ARGS ----

    parser = argparse.ArgumentParser()
    # run args
    parser.add_argument("--run_name", type=str, default="dev")
    parser.add_argument("--config", type=str, default="walker.yml")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to start from")
    parser.add_argument("--total_timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cuda", action="store_true")
    # logging args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_local", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    # sac args
    parser.add_argument("--n_experts", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--nonlinear_actor", action="store_true")
    parser.add_argument("--nonlinear_actor_size", type=str)
    parser.add_argument("--aux_loss", type=str)
    parser.add_argument("--aux_loss_weight", type=float)
    args = parser.parse_args()

    # load config
    cfg = init_cfg(f"config/{args.config}")
    # update config with args
    cfg.update(**vars(args))
    cfg.log.update(**vars(args))
    cfg.sac.update(**vars(args))
    if args.cuda:
        cfg.device = "cuda"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- LOGGING ----

    run_path = f"{args.config.split('.')[0]}/{args.run_name}_{time.strftime('%m%d_%H%M')}"
    cfg.run_path = "runs/" + run_path
    os.makedirs(cfg.run_path, exist_ok=True)

    # save config to run path
    cfg.to_yaml(f"{cfg.run_path}/config.yml")

    if cfg.log.log_local:
        logging.basicConfig(
            filename=f"{cfg.run_path}/log.log",
            level=logging.INFO, format="%(message)s"
        )
        logging.info(
            "global_step,episodic_return,episodic_length"+ \
            "".join([f",episodic_expert_{i}_ratio" for i in range(cfg.sac.n_experts)])
        )
    if cfg.log.wandb:
        wandb.init(project="moe-sac", name=run_path)

    # ---- TRAINING ----

    train(cfg)
