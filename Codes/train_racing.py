import time
import os
import csv
from dataclasses import dataclass
import numpy as np
import torch
from drone_env import QuadRacingEnv
from ppo_agent import PPOAgent, PPOConfig

@dataclass
class TrainConfig:
    total_timesteps: int = 2_000_000 
    log_interval: int = 10
    save_path: str = "ppo_quad_racing.pt"
    init_path: str = "ppo_quad_waypoint.pt" 
    csv_path: str = "racing_training_log.csv" # Log file for graphs
    seed: int = 0

def make_env(seed: int = 0):
    env = QuadRacingEnv(xml_path="drone.xml", seed=seed, max_steps=4000)
    return env

def main():
    cfg = TrainConfig()
    
    ppo_cfg = PPOConfig()
    ppo_cfg.rollout_steps = 8192 
    ppo_cfg.lr = 3e-5 

    env = make_env(seed=cfg.seed)
    obs, info = env.reset(seed=cfg.seed)
    
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], ppo_cfg)

    # --- LOADING LOGIC ---
    if os.path.exists(cfg.save_path):
        print(f"Found existing racing policy: {cfg.save_path}")
        print("   RESUMING TRAINING (Fine-tuning)...")
        state_dict = torch.load(cfg.save_path, map_location=agent.device)
        agent.ac.load_state_dict(state_dict)
    elif os.path.exists(cfg.init_path):
        print(f"No racing policy found. Loading curriculum: {cfg.init_path}")
        print("   STARTING NEW RACING CURRICULUM...")
        state_dict = torch.load(cfg.init_path, map_location=agent.device)
        agent.ac.load_state_dict(state_dict)
    else:
        print("No policies found! Training from scratch (Very Hard Mode).")

    # --- CSV LOGGER SETUP ---
    # We open in 'append' mode if resuming, 'write' if starting new
    file_mode = 'a' if os.path.exists(cfg.csv_path) and os.path.exists(cfg.save_path) else 'w'
    log_file = open(cfg.csv_path, file_mode, newline='')
    logger = csv.writer(log_file)
    
    # Write header only if new file
    if file_mode == 'w':
        logger.writerow(['update', 'steps', 'avg_return', 'laps', 'fps'])
    
    print(f"📄 Logging data to {cfg.csv_path}")

    total_steps = 0
    num_updates = 0
    start_time = time.time()

    while total_steps < cfg.total_timesteps:
        obs_buf, acts_buf, rews_buf, vals_buf, logp_buf, dones_buf = [], [], [], [], [], []
        
        for _ in range(ppo_cfg.rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            obs_buf.append(obs)
            acts_buf.append(action)
            rews_buf.append(reward)
            vals_buf.append(value)
            logp_buf.append(log_prob)
            dones_buf.append(float(terminated or truncated))
            
            obs = next_obs
            total_steps += 1
            
            if terminated or truncated:
                obs, _ = env.reset()
                
            if total_steps >= cfg.total_timesteps: break

        obs_t = torch.as_tensor(np.array(obs_buf, dtype=np.float32))
        act_t = torch.as_tensor(np.array(acts_buf, dtype=np.float32))
        rew_t = torch.as_tensor(np.array(rews_buf, dtype=np.float32))
        val_t = torch.as_tensor(np.array(vals_buf, dtype=np.float32)).to(agent.device)
        logp_t = torch.as_tensor(np.array(logp_buf, dtype=np.float32))
        done_t = torch.as_tensor(np.array(dones_buf, dtype=np.float32))

        with torch.no_grad():
            _, _, last_val = agent.ac.act(torch.as_tensor(obs, dtype=torch.float32).to(agent.device))
            last_val = float(last_val)
            
        adv_t, ret_t = agent.compute_gae(rew_t.to(agent.device), val_t, done_t.to(agent.device), last_val)
        
        batch = {"obs": obs_t, "actions": act_t, "logprobs": logp_t, "returns": ret_t.cpu(), "adv": adv_t.cpu()}
        log_info = agent.update(batch)
        num_updates += 1

        if num_updates % cfg.log_interval == 0:
            ep_count = done_t.sum().item()
            avg_ret = rew_t.sum().item() / ep_count if ep_count > 0 else 0.0
            fps = int(total_steps / (time.time() - start_time))
            laps = info.get('laps', 0)
            
            print(f"Upd {num_updates} | Steps {total_steps} | Ret {avg_ret:.1f} | Laps {laps} | FPS {fps}")

            # Log to CSV
            logger.writerow([num_updates, total_steps, avg_ret, laps, fps])
            log_file.flush()

            torch.save(agent.ac.state_dict(), cfg.save_path)
            print(f"   Saved checkpoint to {cfg.save_path}")

    log_file.close()
    print("Training Complete!")

if __name__ == "__main__":
    main()