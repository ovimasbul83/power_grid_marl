from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam


class RolloutBuffer:
    def __init__(self, n_steps, n_envs, n_agents, obs_dim, action_dim, gamma, gae_lambda):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        S, E, N, D = self.n_steps, self.n_envs, self.n_agents, self.obs_dim
        self.obs  = np.zeros((S, E, N, D), dtype=np.float32)
        self.acts = np.zeros((S, E, N),    dtype=np.float32)
        self.rews = np.zeros((S, E, N),    dtype=np.float32)
        self.dones= np.zeros((S, E),       dtype=np.float32)
        self.vals = np.zeros((S, E, N),    dtype=np.float32)
        self.lps  = np.zeros((S, E, N),    dtype=np.float32)
        self.step = 0

    def add(self, obs, acts, rews, dones, vals, lps):
        self.obs[self.step]  = obs
        self.acts[self.step] = acts.squeeze(-1) if acts.ndim > 2 else acts
        self.rews[self.step] = rews
        self.dones[self.step]= dones
        self.vals[self.step] = vals
        self.lps[self.step]  = lps
        self.step += 1

    def get(self, next_val, device):
        adv = np.zeros_like(self.rews)
        lastgaelam = np.zeros((self.n_envs, self.n_agents))
        for t in reversed(range(self.n_steps)):
            nonterminal = (1.0 - self.dones[t])[:, None]
            nv = next_val if t == self.n_steps-1 else self.vals[t+1]
            delta = self.rews[t] + self.gamma * nv * nonterminal - self.vals[t]
            adv[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
        ret = adv + self.vals
        flat = lambda x: x.reshape(-1)
        return {
            'obs':  torch.FloatTensor(self.obs.reshape(-1, self.obs_dim)).to(device),
            'acts': torch.FloatTensor(flat(self.acts)).unsqueeze(-1).to(device),
            'lps':  torch.FloatTensor(flat(self.lps)).to(device),
            'adv':  torch.FloatTensor(flat(adv)).to(device),
            'ret':  torch.FloatTensor(flat(ret)).to(device),
        }


class MAPPOTrainer:
    def __init__(self, policy, n_agents, obs_dim, action_dim=1,
                 n_steps=256, n_epochs=2, batch_size=512, lr=3e-5,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
                 ent_coef_final=0.001, total_updates=24,
                 gamma=0.99, gae_lambda=0.95,
                 normalize_rewards=True, device='cpu', n_envs=1):
        self.policy=policy.to(device); self.n_agents=n_agents; self.n_envs=n_envs
        self.n_steps=n_steps; self.n_epochs=n_epochs
        self.batch_size=batch_size; self.clip_eps=clip_eps
        self.vf_coef=vf_coef;self.ent_coef_init = ent_coef
        self.ent_coef_final = ent_coef_final
        self.total_updates = total_updates
        self.update_idx = 0
        self.normalize_rewards=normalize_rewards; self.device=device
        self.rew_mean=0.; self.rew_var=0.; self.rew_cnt=0
        self.rew_std=1.0

        actor_params  = [p for n,p in policy.named_parameters() if 'critic' not in n]
        critic_params = [p for n,p in policy.named_parameters() if 'critic'     in n]

        self.actor_opt  = Adam(actor_params, lr=3e-4, eps=1e-5)
        self.critic_opt = Adam(critic_params, lr=1e-3, eps=1e-5)

        self.buf=RolloutBuffer(n_steps,n_envs,n_agents,obs_dim,action_dim,gamma,gae_lambda)

    @torch.no_grad()
    def collect(self, vec_env):
        self.buf.reset()
        obs = vec_env.reset()
        ep_rew     = np.zeros((self.n_envs, self.n_agents))
        ep_raw_rew = np.zeros((self.n_envs, self.n_agents))
        ep_rews=[]; ep_raw_rews=[]; fds=[]

        obs_pinned = torch.empty(self.n_envs * self.n_agents, obs.shape[-1],
                                 pin_memory=True)

        for _ in range(self.n_steps):
            obs_pinned.copy_(torch.from_numpy(obs.reshape(-1, obs.shape[-1])))
            obs_t = obs_pinned.to(self.device, non_blocking=True)
            acts, lps, vals = self.policy.get_action(obs_t)
            acts_np = acts.reshape(self.n_envs, self.n_agents)
            lps_np  = lps.reshape(self.n_envs, self.n_agents).cpu().numpy()
            vals_np = vals.reshape(self.n_envs, self.n_agents).cpu().numpy()

            next_obs, rews, truncs, info = vec_env.step(acts_np)
            ep_raw_rew += rews

            if self.normalize_rewards:
                batch_rews = rews.flatten()
                n = len(batch_rews)
                self.rew_cnt += n
                batch_mean = batch_rews.mean()
                batch_var  = batch_rews.var()
                delta = batch_mean - self.rew_mean
                self.rew_mean += delta * n / self.rew_cnt
                self.rew_var  += batch_var * n + delta**2 * (self.rew_cnt - n) * n / self.rew_cnt
                if self.rew_cnt > 100:
                    self.rew_std = np.sqrt(self.rew_var / self.rew_cnt) + 1e-8
                    rews = np.clip(rews / self.rew_std, -1.0, 1.0)

            self.buf.add(obs, acts_np, rews, truncs.astype(np.float32), vals_np, lps_np)
            ep_rew += rews
            fds.append(info['mean_freq_dev'])

            for i in range(self.n_envs):
                if truncs[i]:
                    ep_rews.append(ep_rew[i].mean())
                    ep_raw_rews.append(ep_raw_rew[i].mean())
                    ep_rew[i]     = 0.
                    ep_raw_rew[i] = 0.
            obs = next_obs

        obs_pinned.copy_(torch.from_numpy(obs.reshape(-1, obs.shape[-1])))
        obs_t = obs_pinned.to(self.device, non_blocking=True)
        _, _, lv = self.policy.get_action(obs_t)
        next_val = lv.reshape(self.n_envs, self.n_agents).cpu().numpy()
        batch = self.buf.get(next_val, self.device)
        raw_rew = np.mean(ep_raw_rews) if ep_raw_rews else ep_raw_rew.mean()
        return batch, (np.mean(ep_rews) if ep_rews else ep_rew.mean()), np.mean(fds), raw_rew

    def update(self, batch):
        obs,acts,olp,adv,ret = batch['obs'],batch['acts'],batch['lps'],batch['adv'],batch['ret']
        n=obs.shape[0]; idx=np.arange(n); pg_l,vf_l,gnorm_l,ent_l=[],[],[],[]

        actor_params  = [p for nm,p in self.policy.named_parameters() if 'critic' not in nm]
        critic_params = [p for nm,p in self.policy.named_parameters() if 'critic'     in nm]

        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for s in range(0,n,self.batch_size):
                mb=idx[s:s+self.batch_size]

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
                                adv[mb]*ratio.clamp(1-self.clip_eps,1+self.clip_eps)).mean()
                (pg - self.ent_coef * ent.mean()).backward()
                gnorm=nn.utils.clip_grad_norm_(actor_params, 1.0)
                self.actor_opt.step()

                pg_l.append(pg.item()); vf_l.append(vf.item())
                gnorm_l.append(gnorm.item()); ent_l.append(ent.mean().item())

        return {'pg':np.mean(pg_l),'vf':np.mean(vf_l),
                'gnorm':np.mean(gnorm_l),'ent':np.mean(ent_l)}

    def train_step(self, env):
        self.update_idx += 1
        progress = min(1.0, self.update_idx / self.total_updates)
        self.ent_coef = self.ent_coef_init + progress * (self.ent_coef_final - self.ent_coef_init)

        batch, mr, mfd, raw_rew = self.collect(env)
        losses = self.update(batch)
        losses["ent_coef"] = self.ent_coef
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
