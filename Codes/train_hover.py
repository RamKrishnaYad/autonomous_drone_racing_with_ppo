# train_hover.py
import time
from dataclasses import dataclass

import numpy as np
import torch

from drone_env import QuadHoverEnv
from ppo_agent import PPOAgent, PPOConfig


@dataclass
class TrainConfig:
    total_timesteps: int = 400_000
    log_interval: int = 10      # PPO updates between logs
    save_path: str = "ppo_quad_hover.pt"
    seed: int = 0


def make_env(seed: int = 0) -> QuadHoverEnv:
    env = QuadHoverEnv(xml_path="drone.xml", seed=seed)
    return env


def main():
    cfg = TrainConfig()
    ppo_cfg = PPOConfig()

    env = make_env(seed=cfg.seed)
    obs, info = env.reset(seed=cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim, act_dim, ppo_cfg)

    total_steps = 0
    num_updates = 0

    start_time = time.time()

    while total_steps < cfg.total_timesteps:
        # Rollout storage
        obs_buf = []
        actions_buf = []
        logprobs_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []

        # Collect rollout
        for _ in range(ppo_cfg.rollout_steps):
            action, log_prob, value = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_buf.append(obs.copy())
            actions_buf.append(action.copy())
            logprobs_buf.append(log_prob)
            rewards_buf.append(reward)
            dones_buf.append(float(done))
            values_buf.append(value)

            obs = next_obs
            total_steps += 1

            if done:
                obs, info = env.reset()

            if total_steps >= cfg.total_timesteps:
                break

        # Convert buffers to tensors
        obs_t = torch.as_tensor(np.array(obs_buf, dtype=np.float32))
        actions_t = torch.as_tensor(np.array(actions_buf, dtype=np.float32))
        logprobs_t = torch.as_tensor(np.array(logprobs_buf, dtype=np.float32))
        rewards_t = torch.as_tensor(np.array(rewards_buf, dtype=np.float32))
        dones_t = torch.as_tensor(np.array(dones_buf, dtype=np.float32))
        values_t = torch.as_tensor(np.array(values_buf, dtype=np.float32), device=agent.device)

        # Bootstrap value for last obs
        with torch.no_grad():
            last_value = 0.0
            if len(obs_buf) > 0:
                last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
                _, _, last_v = agent.ac.act(last_obs_t)
                last_value = float(last_v)

        # Compute advantages and returns using GAE
        adv_t, returns_t = agent.compute_gae(
            rewards=rewards_t.to(agent.device),
            values=values_t,
            dones=dones_t.to(agent.device),
            last_value=last_value,
        )

        batch = {
            "obs": obs_t,
            "actions": actions_t,
            "logprobs": logprobs_t,
            "returns": returns_t.cpu(),
            "adv": adv_t.cpu(),
        }

        # PPO update
        log_info = agent.update(batch)
        num_updates += 1

        # Logging
        if num_updates % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            fps = int(total_steps / elapsed) if elapsed > 0 else 0
            avg_ep = max(1.0, dones_t.sum().item())
            avg_ret = float(rewards_t.sum().item() / avg_ep)
            print(
                f"Update {num_updates} | Steps {total_steps} | "
                f"AvgEpRet {avg_ret:.2f} | "
                f"PolicyLoss {log_info['policy_loss']:.3f} | "
                f"ValueLoss {log_info['value_loss']:.3f} | "
                f"Entropy {log_info['entropy']:.3f} | "
                f"KL {log_info['approx_kl']:.4f} | "
                f"FPS {fps}"
            )

    # Save trained policy
    torch.save(agent.ac.state_dict(), cfg.save_path)
    print(f"Saved hover policy to {cfg.save_path}")


if __name__ == "__main__":
    main()
