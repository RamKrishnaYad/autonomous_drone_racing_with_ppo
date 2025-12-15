import numpy as np
import torch
import imageio
import mujoco

from drone_env import QuadRacingEnv
from ppo_agent import ActorCritic

def load_actor(obs_dim, act_dim, path, device):
    ac = ActorCritic(obs_dim, act_dim).to(device)
    state_dict = torch.load(path, map_location=device)
    ac.load_state_dict(state_dict)
    ac.eval()
    return ac

def run_episode(env, ac, device, render=False, save_gif_path=None):
    obs, info = env.reset()
    done = False
    ep_ret = 0
    frames = []
    
    renderer = None
    if save_gif_path:
        renderer = mujoco.Renderer(env.model, height=480, width=640)

    # Run until done (which is now 3 laps)
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _, _ = ac.act(obs_t)
            
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated or truncated
        ep_ret += reward
        
        if renderer:
            renderer.update_scene(env.data)
            frames.append(renderer.render())
        
        if render:
            env.render()

    if save_gif_path and frames:
        imageio.mimsave(save_gif_path, frames, fps=30)
        print(f"Saved GIF: {save_gif_path}")
        
    return ep_ret, info.get('laps', 0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Increase max_steps to allow full 3 laps
    env = QuadRacingEnv(xml_path="drone.xml", max_steps=4000)
    
    model_path = "ppo_quad_racing.pt"
    ac = load_actor(12, 4, model_path, device)
    
    print("Evaluating Racing Policy (3 Laps)...")
    for i in range(2):
        ret, laps = run_episode(env, ac, device, render=True, save_gif_path=f"race_laps_{i}.gif")
        print(f"Run {i}: Return {ret:.1f}, Laps Completed: {laps}")

if __name__ == "__main__":
    main()