import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple
from stable_baselines3.common.buffers import ReplayBuffer
import math


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, cfg, env):
        super().__init__()

        self.cfg = cfg.sac

        observation_shape = np.prod(env.single_observation_space.shape)
        action_shape = np.prod(env.single_action_space.shape)

        self.router = nn.Linear(observation_shape, self.cfg.n_experts)
        self.topk = self.cfg.topk

        self.mean_experts = nn.ModuleList([nn.Linear(observation_shape, action_shape) for _ in range(self.cfg.n_experts)])
        self.log_std_experts = nn.ModuleList([nn.Linear(observation_shape, action_shape) for _ in range(self.cfg.n_experts)])

        self.register_buffer("episodic_expert_count", torch.zeros(self.cfg.n_experts, dtype=int))  # for logging

        self.register_buffer("router_importance", torch.zeros(self.cfg.n_experts))
        self.register_buffer("router_load", torch.zeros(self.cfg.n_experts))

        # action scaling and bias
        self.register_buffer("action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32))

        self.noise_distr = torch.distributions.Normal(
            loc=torch.tensor([0.0]*self.cfg.n_experts), scale=torch.tensor([1.0/self.cfg.n_experts]*self.cfg.n_experts)
        )

    def forward(self, x, router_noise):
        x = x.float()

        router_logits = self.router(x)

        if router_noise:
            noisy_logits = router_logits + self.noise_distr.sample()

            importance = F.softmax(noisy_logits, dim=-1).sum(0)
            self.router_importance = (torch.std(importance)/torch.mean(importance))**2

            threshold = torch.max(noisy_logits, dim=-1).values
            load = (1 - self.noise_distr.cdf(threshold.unsqueeze(1) - router_logits)).sum(0)
            self.router_load = (torch.std(load)/torch.mean(load))**2

            router_logits = noisy_logits

        self.router_probs = F.softmax(router_logits, dim=-1)

        topk_router_probs, topk_router_indices = torch.topk(self.router_probs, self.topk, dim=-1)
        sparse_router_probs = torch.zeros_like(self.router_probs).scatter_(index=topk_router_indices, src=topk_router_probs, dim=-1)

        if x.shape[0] == 1:
            self.episodic_expert_count[topk_router_indices] += 1

        expert_outputs = torch.stack([expert(x) for expert in self.mean_experts], dim=1)
        log_std_outputs = torch.stack([expert(x) for expert in self.log_std_experts], dim=1)

        mean = torch.sum(expert_outputs * sparse_router_probs.unsqueeze(-1), dim=1)
        log_std = torch.sum(log_std_outputs * sparse_router_probs.unsqueeze(-1), dim=1)

        # apply action scaling and bias
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x, router_noise=False):
        mean, log_std = self(x, router_noise)
        std = log_std.exp()
        x_t = torch.randn_like(mean) * std + mean  # faster than torch.distributions.Normal(mean, std).rsample()
        y_t = torch.tanh(x_t)  # scale to -1, 1
        action = y_t * self.action_scale + self.action_bias  # scale to environment's range
        log_prob = -0.5 * ((x_t - mean) / std).pow(2) - std.log() - 0.5 * math.log(2 * math.pi)  # gaussian log likelihood
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)  # adjustment for tanh
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias  # scale to environment's range
        return action, log_prob, mean


class SoftQNetwork(nn.Module):
    def __init__(self, cfg, env):
        super().__init__()

        input_dim = np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape)
        in_dims = [input_dim] + [256] * cfg.sac.q_depth
        out_dims = [256] * cfg.sac.q_depth + [1]

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)])

    def forward(self, x, a):
        x = x.float()
        x = torch.cat([x, a], 1)

        for fc_layer in self.fc_list[:-1]:
            x = F.relu(fc_layer(x))
        x = self.fc_list[-1](x)

        return x

SACComponents = namedtuple("SACComponents", ["actor", "qf1", "qf2", "qf1_target", "qf2_target", "q_optimizer", "actor_optimizer", "rb", "target_entropy", "log_alpha", "a_optimizer", "counter"])

def setup_sac(cfg, env):
    actor = Actor(cfg, env).to(cfg.device)
    qf1 = SoftQNetwork(cfg, env).to(cfg.device)
    qf2 = SoftQNetwork(cfg, env).to(cfg.device)
    qf1_target = SoftQNetwork(cfg, env).to(cfg.device)
    qf2_target = SoftQNetwork(cfg, env).to(cfg.device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.sac.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=cfg.sac.policy_lr)

    if cfg.sac.alpha_auto == True:
        target_entropy = -torch.prod(torch.tensor(env.single_action_space.shape).to(cfg.device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=cfg.device)
        a_optimizer = optim.Adam([log_alpha], lr=cfg.sac.q_lr)
    else:
        target_entropy = None
        log_alpha = None
        a_optimizer = None

    # MinMax Replay Buffer so we can add new best or worst experiences
    rb = ReplayBuffer(
        cfg.sac.buffer_size,
        env.single_observation_space,
        env.single_action_space,
        cfg.device,
        handle_timeout_termination=False,
    )

    counter = {'n_steps': 0}

    return SACComponents(actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, rb, target_entropy, log_alpha, a_optimizer, counter)

def train_sac(cfg, sac):
    alpha = sac.log_alpha.exp().item() if cfg.sac.alpha_auto else cfg.sac.alpha

    data = sac.rb.sample(cfg.sac.batch_size)
    with torch.no_grad():
        next_state_actions, next_state_log_pi, _ = sac.actor.get_action(data.next_observations)
        qf1_next_target = sac.qf1_target(data.next_observations, next_state_actions)
        qf2_next_target = sac.qf2_target(data.next_observations, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        next_q_value = data.rewards.view(-1) + (1 - data.dones.view(-1)) * cfg.sac.gamma * (min_qf_next_target).view(-1)

    qf1_a_values = sac.qf1(data.observations, data.actions).view(-1)
    qf2_a_values = sac.qf2(data.observations, data.actions).view(-1)
    qf_loss = F.mse_loss(qf1_a_values, next_q_value) + F.mse_loss(qf2_a_values, next_q_value)

    # optimize the model
    sac.q_optimizer.zero_grad()
    qf_loss.backward()
    sac.q_optimizer.step()

    if sac.counter['n_steps'] % cfg.sac.policy_frequency == 0:  # TD 3 Delayed update support
        for _ in range(cfg.sac.policy_frequency):  # compensate for the delay in policy updates
            pi, log_pi, _ = sac.actor.get_action(data.observations, router_noise=True)
            qf1_pi = sac.qf1(data.observations, pi)
            qf2_pi = sac.qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

            aux_loss = 0.5 * sac.actor.router_importance + 0.5 * sac.actor.router_load
            actor_loss += 0.01 * aux_loss

            sac.actor_optimizer.zero_grad()
            actor_loss.backward()
            sac.actor_optimizer.step()

            if cfg.sac.alpha_auto == True:
                with torch.no_grad():
                    _, log_pi, _ = sac.actor.get_action(data.observations)
                alpha_loss = (-sac.log_alpha.exp() * (log_pi + sac.target_entropy)).mean()

                sac.a_optimizer.zero_grad()
                alpha_loss.backward()
                sac.a_optimizer.step()
                alpha = sac.log_alpha.exp().item()

    if sac.counter['n_steps'] % cfg.sac.target_network_frequency == 0:
        for param, target_param in zip(sac.qf1.parameters(), sac.qf1_target.parameters()):
            target_param.data.copy_(cfg.sac.tau * param.data + (1 - cfg.sac.tau) * target_param.data)
        for param, target_param in zip(sac.qf2.parameters(), sac.qf2_target.parameters()):
            target_param.data.copy_(cfg.sac.tau * param.data + (1 - cfg.sac.tau) * target_param.data)

    sac.counter['n_steps']  += 1
