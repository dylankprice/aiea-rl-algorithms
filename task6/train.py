import gymnasium as gym
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from model import ActorCritic


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")


def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array",
                   lap_complete_percent=0.95,
                   domain_randomize=False, continuous=True)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.GrayscaleObservation(env)        # native 96x96, no resize
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


def obs_to_tensor(obs):
    return torch.from_numpy(np.array(obs) / 255.).float().to(device)


def collect_rollout(envs, actorcritic, current_obs, T, N):
    buf_obs   = torch.zeros((T, N, 4, 96, 96), device=device)
    buf_acts  = torch.zeros((T, N, 3), device=device)  # tanh actions, stored for evaluate()
    buf_lps   = torch.zeros((T, N), device=device)
    buf_vals  = torch.zeros((T+1, N), device=device)
    buf_rews  = torch.zeros((T, N), device=device)
    buf_dones = torch.zeros((T, N), device=device)

    with torch.no_grad():
        for t in range(T):
            buf_obs[t] = current_obs

            # action_env -> rescaled for env (gas/brake in [0,1])
            # action     -> raw tanh output, stored for evaluate()
            action_env, action, log_prob, value = actorcritic.get_action(current_obs)

            obs_np, reward, term, trunc, _ = envs.step(action_env.cpu().numpy())
            done = term | trunc

            current_obs = obs_to_tensor(obs_np)

            buf_acts[t]  = action        # store raw tanh, NOT action_env
            buf_lps[t]   = log_prob
            buf_vals[t]  = value
            buf_rews[t]  = torch.tensor(reward, dtype=torch.float32, device=device)  # scale rewards
            buf_dones[t] = torch.tensor(done, dtype=torch.float32, device=device)

        # last value for GAE bootstrap
        _, _, _, last_val = actorcritic.get_action(current_obs)
        buf_vals[T] = last_val

    return buf_obs, buf_acts, buf_lps, buf_vals, buf_rews, buf_dones, current_obs


def compute_advantages(buf_rews, buf_vals, buf_dones, T, gamma, gae_lambda):
    advantages = torch.zeros_like(buf_rews)
    gae = 0.
    with torch.no_grad():
        for t in reversed(range(T)):
            nd    = 1. - buf_dones[t]
            delta = buf_rews[t] + gamma * buf_vals[t+1] * nd - buf_vals[t]
            gae   = delta + gamma * gae_lambda * nd * gae
            advantages[t] = gae
    return advantages


def train(envs, actorcritic,
          T=256, K=4, batch_size=256,
          gamma=0.99, gae_lambda=0.95,
          vf_coeff=0.5, ent_coeff=0.02,
          clip_eps=0.2, nb_iterations=1500,
          lr=3e-4):

    optimizer   = torch.optim.Adam(actorcritic.parameters(), lr=lr, eps=1e-5)
    scheduler   = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1., end_factor=0., total_iters=nb_iterations)
    writer      = SummaryWriter()
    global_step = 0
    N           = envs.num_envs
    best_reward = -np.inf

    obs_np, _   = envs.reset()
    current_obs = obs_to_tensor(obs_np)

    
    for iteration in range(nb_iterations):
        buf_obs, buf_acts, buf_lps, buf_vals, buf_rews, buf_dones, current_obs = \
            collect_rollout(envs, actorcritic, current_obs, T, N)

        advantages = compute_advantages(buf_rews, buf_vals, buf_dones, T, gamma, gae_lambda)

        current_ent_coeff = ent_coeff * max(0.1, 1 - iteration / nb_iterations)

        flat_obs  = buf_obs.reshape(-1, 4, 96, 96)
        flat_acts = buf_acts.reshape(-1, 3)
        flat_lps  = buf_lps.reshape(-1)
        flat_vals = buf_vals[:T].reshape(-1)
        flat_adv  = advantages.reshape(-1)
        flat_adv  = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        loader = DataLoader(
            TensorDataset(flat_adv, flat_obs, flat_acts, flat_lps, flat_vals),
            batch_size=batch_size, shuffle=True)

        for _ in range(K):
            for b_adv, b_obs, b_act, b_old_lp, b_old_val in loader:
                log_prob, value, entropy = actorcritic.evaluate(b_obs, b_act)
                ratio   = torch.exp(log_prob - b_old_lp)
                returns = b_adv + b_old_val

                p_loss = -torch.min(
                    ratio * b_adv,
                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                ).mean()

                v_clip = b_old_val + torch.clamp(value - b_old_val, -clip_eps, clip_eps)
                v_loss = torch.max(
                    F.mse_loss(returns, value, reduction='none'),
                    F.mse_loss(returns, v_clip, reduction='none')
                ).mean()

                loss = p_loss + vf_coeff * v_loss - current_ent_coeff * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actorcritic.parameters(), 0.5)
                optimizer.step()

                writer.add_scalar("Loss/policy",  p_loss.item(), global_step)
                writer.add_scalar("Loss/value",   v_loss.item(), global_step)
                writer.add_scalar("Loss/entropy", entropy.mean().item(), global_step)
                global_step += 1

        episode_reward = buf_rews.sum(0).mean().item()
        writer.add_scalar("Reward/iteration", episode_reward, iteration)
        writer.flush()

        # log mean actions to verify gas/brake are in [0,1] and steering in [-1,1]
        writer.add_scalar("Actions/steer", buf_acts[:, :, 0].mean().item(), iteration)
        writer.add_scalar("Actions/gas",   buf_acts[:, :, 1].mean().item(), iteration)
        writer.add_scalar("Actions/brake", buf_acts[:, :, 2].mean().item(), iteration)

        print(f"Iteration {iteration}/{nb_iterations} | Reward: {episode_reward:.2f} | Best: {best_reward:.2f}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(actorcritic.state_dict(), "racecar_ppo_model.pt")

        scheduler.step()

    writer.close()


if __name__ == "__main__":
    envs        = gym.vector.AsyncVectorEnv([make_env for _ in range(8)])
    actorcritic = ActorCritic(envs.single_action_space.shape[0]).to(device)  # shape[0]=3

    train(envs, actorcritic)
    envs.close()