import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(input_dim: int, output_dim: int, hidden_sizes=(64, 64), activation=nn.Tanh) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Gaussian policy (actor) + value function (critic).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.actor_mu = mlp(obs_dim, act_dim, hidden_sizes)
        # log_std as independent parameter (diagonal Gaussian)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic = mlp(obs_dim, 1, hidden_sizes)

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mu = self.actor_mu(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        dist = self.get_dist(obs)
        value = self.critic(obs).squeeze(-1)  # (batch,)
        return dist, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: (obs_dim,) or (batch, obs_dim)
        returns: action, log_prob, value
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        dist, value = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used during PPO update.
        obs: (batch, obs_dim)
        actions: (batch, act_dim)
        returns: log_probs, entropy, values
        """
        dist, value = self.forward(obs)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.001 
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 1e-4
    train_iters: int = 10          # PPO epochs per update
    batch_size: int = 128           # mini-batch size
    rollout_steps: int = 8192      # steps per update


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.lr)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        obs: (obs_dim,) numpy
        returns: action (np, act_dim), log_prob (float), value (float)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_t, log_prob_t, value_t = self.ac.act(obs_t)
        return (
            action_t.cpu().numpy(),
            log_prob_t.item(),
            value_t.item(),
        )

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rewards: (T,)
        values: (T,)
        dones: (T,) with 1.0 if done, 0.0 otherwise
        last_value: scalar V(s_T)
        returns: advantages (T,), returns (T,)
        """
        T = rewards.shape[0]
        adv = torch.zeros(T, device=self.device)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.config.gamma * self.config.lam * next_nonterminal * last_gae
            adv[t] = last_gae

        returns = adv + values
        return adv, returns

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        batch has keys:
          'obs'      : (N, obs_dim)
          'actions'  : (N, act_dim)
          'logprobs' : (N,)
          'returns'  : (N,)
          'adv'      : (N,)
        """
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_logprobs = batch["logprobs"].to(self.device)
        returns = batch["returns"].to(self.device)
        adv = batch["adv"].to(self.device)

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = obs.size(0)
        batch_size = self.config.batch_size
        train_iters = self.config.train_iters

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0

        for _ in range(train_iters):
            # random minibatch indices
            idx = torch.randperm(N)
            for start in range(0, N, batch_size):
                end = start + batch_size
                mb_idx = idx[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = adv[mb_idx]

                new_logprobs, entropy, values = self.ac.evaluate_actions(mb_obs, mb_actions)

                logratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(logratio)

                # PPO clipped surrogate objective (Schulman et al. 2017)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value function loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.config.vf_coef * value_loss + self.config.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().abs().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += approx_kl
                num_updates += 1

        log_dict = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_kl / num_updates,
        }
        return log_dict
