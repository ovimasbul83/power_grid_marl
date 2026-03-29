from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam


class RolloutBuffer:
    def __init__(self, T, N, obs_dim, act_dim, gamma=0.99, lam=0.95):
        self.T = T; self.N = N; self.obs_dim = obs_dim; self.act_dim = act_dim
        self.gamma = gamma; self.lam = lam; self.reset()

    def reset(self):
        T, N = self.T, self.N
        self.obs   = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.acts  = np.zeros((T, N, self.act_dim), dtype=np.float32)
        self.rews  = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.vals  = np.zeros((T, N), dtype=np.float32)
        self.lps   = np.zeros((T, N), dtype=np.float32)
        self.ptr   = 0

    def add(self, obs, acts, rews, dones, vals, lps):
        self.obs[self.ptr]=obs; self.acts[self.ptr]=acts
        self.rews[self.ptr]=rews; self.dones[self.ptr]=dones
        self.vals[self.ptr]=vals; self.lps[self.ptr]=lps; self.ptr+=1

    def get(self, last_vals, device):
        adv = np.zeros_like(self.rews); gae = np.zeros(self.N)
        for t in reversed(range(self.T)):
            nv = last_vals if t == self.T-1 else self.vals[t+1]
            mask = 1. - self.dones[t]
            delta = self.rews[t] + self.gamma*nv*mask - self.vals[t]
            gae = delta + self.gamma*self.lam*mask*gae; adv[t] = gae
        ret = adv + self.vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        f = lambda x: torch.FloatTensor(x).to(device)
        return {'obs':  f(self.obs.reshape(-1, self.obs_dim)),
                'acts': f(self.acts.reshape(-1, self.act_dim)),
                'lps':  f(self.lps.reshape(-1)),
                'adv':  f(adv.reshape(-1)),
                'ret':  f(ret.reshape(-1))}


class MAPPOTrainer:
    def __init__(self, policy, n_agents, obs_dim, action_dim=1,
                 n_steps=256, n_epochs=2, batch_size=2048,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.05,
                 ent_coef_final=0.005, total_updates=24,
                 gamma=0.99, gae_lambda=0.95,
                 normalize_rewards=True, device='cpu'):
        self.policy = policy.to(device); self.n_agents = n_agents
        self.n_steps = n_steps; self.n_epochs = n_epochs
        self.batch_size = batch_size; self.clip_eps = clip_eps
        self.vf_coef = vf_coef; self.ent_coef = self.ent_coef_init = ent_coef
        self.ent_coef_final = ent_coef_final; self.total_updates = total_updates
        self.update_idx = 0; self.normalize_rewards = normalize_rewards
        self.device = device
        self.rew_mean = 0.; self.rew_var = 0.; self.rew_cnt = 0

        actor_params  = [p for n, p in policy.named_parameters() if 'critic' not in n]
        critic_params = [p for n, p in policy.named_parameters() if 'critic' in n]
        self.actor_opt  = Adam(actor_params,  lr=1e-3, eps=1e-5)
        self.critic_opt = Adam(critic_params, lr=3e-4, eps=1e-5)
        self.buf = RolloutBuffer(n_steps, n_agents, obs_dim, action_dim, gamma, gae_lambda)

    @torch.no_grad()
    def collect(self, env):
        self.buf.reset(); obs, _ = env.reset()
        ep_rew = np.zeros(self.n_agents); ep_raw_rew = np.zeros(self.n_agents)
        ep_rews, ep_raw_rews, fds = [], [], []
        for _ in range(self.n_steps):
            obs_t = torch.FloatTensor(obs).to(self.device)
            acts, lps, vals = self.policy.get_action(obs_t)
            next_obs, rews, _, trunc, info = env.step(acts.squeeze(-1))
            ep_raw_rew += rews
            if self.normalize_rewards:
                for r in rews.flatten():
                    self.rew_cnt += 1
                    delta = r - self.rew_mean
                    self.rew_mean += delta / self.rew_cnt
                    self.rew_var  += delta * (r - self.rew_mean)
                if self.rew_cnt > 100:
                    std = np.sqrt(self.rew_var / self.rew_cnt) + 1e-8
                    rews = np.clip(rews / std, -1.0, 1.0)
            dones = np.full(self.n_agents, trunc, dtype=np.float32)
            self.buf.add(obs, acts, rews, dones, vals.cpu().numpy(), lps.cpu().numpy())
            ep_rew += rews; fds.append(info['mean_freq_dev']); obs = next_obs
            if trunc:
                ep_rews.append(ep_rew.mean()); ep_raw_rews.append(ep_raw_rew.mean())
                ep_rew = np.zeros(self.n_agents); ep_raw_rew = np.zeros(self.n_agents)
                obs, _ = env.reset()
        obs_t = torch.FloatTensor(obs).to(self.device)
        _, _, lv = self.policy.get_action(obs_t)
        batch = self.buf.get(lv.cpu().numpy(), self.device)
        raw_rew = np.mean(ep_raw_rews) if ep_raw_rews else ep_raw_rew.mean()
        return batch, (np.mean(ep_rews) if ep_rews else ep_rew.mean()), np.mean(fds), raw_rew

    def update(self, batch):
        obs, acts, olp, adv, ret = (
            batch['obs'], batch['acts'], batch['lps'], batch['adv'], batch['ret'])
        n = obs.shape[0]; idx = np.arange(n)
        pg_l, vf_l, gnorm_l, ent_l = [], [], [], []
        actor_params  = [p for nm, p in self.policy.named_parameters() if 'critic' not in nm]
        critic_params = [p for nm, p in self.policy.named_parameters() if 'critic' in nm]
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.batch_size):
                mb = idx[s:s+self.batch_size]
                self.critic_opt.zero_grad()
                val = self.policy.evaluate_critic(obs[mb])
                vf  = F.mse_loss(val, ret[mb])
                vf.backward()
                nn.utils.clip_grad_norm_(critic_params, 1.0)
                self.critic_opt.step()
                self.actor_opt.zero_grad()
                nlp, ent = self.policy.evaluate(obs[mb], acts[mb])
                ratio = torch.exp(nlp - olp[mb])
                pg = -torch.min(adv[mb]*ratio,
                                adv[mb]*ratio.clamp(1-self.clip_eps, 1+self.clip_eps)).mean()
                (pg - self.ent_coef * ent.mean()).backward()
                gnorm = nn.utils.clip_grad_norm_(actor_params, 1.0)
                self.actor_opt.step()
                pg_l.append(pg.item()); vf_l.append(vf.item())
                gnorm_l.append(gnorm.item()); ent_l.append(ent.mean().item())
        return {'pg': np.mean(pg_l), 'vf': np.mean(vf_l),
                'gnorm': np.mean(gnorm_l), 'ent': np.mean(ent_l)}

    def train_step(self, env):
        self.update_idx += 1
        progress = min(1.0, self.update_idx / self.total_updates)
        self.ent_coef = self.ent_coef_init + progress * (self.ent_coef_final - self.ent_coef_init)
        batch, mr, mfd, raw_rew = self.collect(env)
        losses = self.update(batch)
        losses['ent_coef'] = self.ent_coef
        return mr, mfd, losses, raw_rew


def train(env, policy, total_steps=200_000, n_steps=1024, device='cpu', verbose=True):
    n_updates = total_steps // n_steps
    trainer = MAPPOTrainer(policy, env.n_generators, env.obs_dim,
                           n_steps=n_steps, normalize_rewards=True, device=device,
                           ent_coef=0.05, ent_coef_final=0.005, total_updates=n_updates)
    rews, fds = [], []
    best_mfd = float('inf'); best_state = None; best_update = -1
    for u in range(1, n_updates+1):
        mr, mfd, losses, raw_rew = trainer.train_step(env)
        rews.append(mr); fds.append(mfd)
        if mfd < best_mfd:
            best_mfd = mfd; best_state = deepcopy(policy.state_dict()); best_update = u
        if verbose and u % max(1, n_updates//10) == 0:
            print(f'  [{u:4d}/{n_updates}] freq_dev={np.mean(fds[-20:]):.5f} '
                  f'raw_rew={raw_rew:.4f} pg={losses["pg"]:.6f} ent={losses["ent"]:.4f}')
    if best_state is not None:
        policy.load_state_dict(best_state)
        if verbose: print(f'  best checkpoint: update {best_update}, mfd={best_mfd:.5f}')
    return policy, rews, fds
