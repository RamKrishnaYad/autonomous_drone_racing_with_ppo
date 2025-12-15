
import numpy as np
import torch
import imageio
import mujoco

from drone_env import QuadWaypointEnv
from ppo_agent import ActorCritic


def load_actor(obs_dim, act_dim, path, device):
    ac = ActorCritic(obs_dim, act_dim).to(device)
    state_dict = torch.load(path, map_location=device)
    ac.load_state_dict(state_dict)
    ac.eval()
    return ac


def run_episode(
    env,
    actor_critic,
    device,
    render: bool = False,
    save_gif_path: str | None = None,
    fps: int = 30,
    max_steps: int = 500,
):
    """
    Run a single evaluation episode.
    If save_gif_path is not None, record frames and save as a GIF at the end.
    """
    obs, info = env.reset()
    done = False
    ep_ret = 0.0
    ep_len = 0

    # For debugging waypoint behavior
    target = info.get("p_target", None)
    if target is not None:
        print(f"Target: {target}")

    # --- GIF recording setup ---
    frames = []
    renderer = None
    if save_gif_path is not None:
        # You can change height/width if you want a different resolution
        renderer = mujoco.Renderer(env.model, height=480, width=640)

    dists = []  # optional: track distance over time

    while not done and ep_len < max_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            # Adjust this call to match your actor_critic API
            # (this matches the earlier ac.act(...) pattern)
            mean_action, _, _ = actor_critic.act(obs_t)
        action = mean_action.cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_ret += reward
        ep_len += 1

        # Track distance to target if provided
        dist = info.get("dist_to_target", None)
        if dist is not None:
            dists.append(dist)

        # --- Capture frame for GIF ---
        if renderer is not None:
            renderer.update_scene(env.data)
            frame = renderer.render()  # RGB array (H, W, 3)
            frames.append(frame)

        # Optional live viewer
        if render:
            env.render()

    # --- Save GIF if requested ---
    if save_gif_path is not None and len(frames) > 0:
        imageio.mimsave(save_gif_path, frames, fps=fps)
        print(f"Saved GIF to: {save_gif_path}")

    # Debug print distances
    if dists:
        print("Distances over time:", [f"{d:.2f}" for d in dists[:20]])

    return ep_ret, ep_len


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = QuadWaypointEnv(xml_path="drone.xml", max_steps=200)

    obs, info = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model_path = "ppo_quad_waypoint.pt"
    ac = load_actor(obs_dim, act_dim, model_path, device)

    print("Evaluating deterministic (mean) waypoint policy...")

    num_episodes = 5
    returns = []
    lengths = []

    for i in range(num_episodes):
        ep_ret, ep_len = run_episode(env, ac, device, render=False)
        returns.append(ep_ret)
        lengths.append(ep_len)
        print(f"Episode {i+1}: return = {ep_ret:.2f}, length = {ep_len}")
        gif_path = f"waypoint_eval_ep{i+1}.gif"
        ep_ret, ep_len = run_episode(env, ac, device,
                                 render=False,
                                 save_gif_path=gif_path,
                                 fps=30)
        print(f"Episode {i+1}: return={ep_ret:.2f}, len={ep_len}")

    print()
    print(f"Average return over {num_episodes} episodes: {np.mean(returns):.2f}")
    print(f"Average episode length: {np.mean(lengths):.1f}")

    env.close()


if __name__ == "__main__":
    main()
