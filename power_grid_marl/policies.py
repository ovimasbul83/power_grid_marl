import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from .gnn import GCNLayer, GATLayer, DynamicGraphBuilder

LOG_STD_MIN, LOG_STD_MAX = -2, 1.0


def mlp(in_d, hid_d, out_d, layers=2):
    mods, d = [], in_d
    for _ in range(layers):
        mods += [nn.Linear(d, hid_d), nn.LayerNorm(hid_d), nn.ReLU()]
        d = hid_d
    mods.append(nn.Linear(d, out_d))
    return nn.Sequential(*mods)


class BaselinePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim=1, hidden_dim=128):
        super().__init__()
        self.actor_encoder  = mlp(obs_dim, hidden_dim, hidden_dim)
        self.actor_mu       = nn.Linear(hidden_dim, action_dim)
        self.actor_std      = nn.Parameter(torch.zeros(action_dim))
        self.critic_encoder = mlp(obs_dim, hidden_dim, hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, obs):
        ha  = F.relu(self.actor_encoder(obs))
        hc  = F.relu(self.critic_encoder(obs))
        mu  = self.actor_mu(ha) * 0.1
        std = self.actor_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        return mu, std, self.critic(hc).squeeze(-1)

    @torch.no_grad()
    def get_action(self, obs):
        mu, std, val = self.forward(obs)
        d = Normal(mu, std); a = d.sample()
        return a.cpu().numpy(), d.log_prob(a).sum(-1), val

    def evaluate(self, obs, actions):
        ha  = F.relu(self.actor_encoder(obs))
        mu  = self.actor_mu(ha) * 0.1
        std = self.actor_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        d   = Normal(mu, std)
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1)

    def evaluate_critic(self, obs):
        hc = F.relu(self.critic_encoder(obs))
        return self.critic(hc).squeeze(-1)


class GNNPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim=1, hidden_dim=128,
                 n_agents=4, model='gat', n_layers=2, top_k=3, grid_adj=None):
        super().__init__()
        self.n_agents = n_agents; self.model = model; self.grid_adj = grid_adj
        self.actor_encoder  = mlp(obs_dim, hidden_dim, hidden_dim)
        self.critic_encoder = mlp(obs_dim, hidden_dim, hidden_dim)
        self.dyn_graph = DynamicGraphBuilder(hidden_dim, top_k) if model == 'dynamic' else None
        self.gnns = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim) if model == 'gcn'
            else GATLayer(hidden_dim, hidden_dim, n_heads=4)
            for _ in range(n_layers)])
        self.actor_mu  = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))
        self.critic    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.graph_alpha = 0.0

    def set_graph_alpha(self, alpha):
        self.graph_alpha = float(np.clip(alpha, 0.0, 1.0))

    def _get_graph(self, hb):
        if self.model != 'dynamic':
            return self.grid_adj.to(hb.device), None
        if self.graph_alpha < 1.0:
            return self.grid_adj.to(hb.device), None
        dyn_ei, dyn_ew = self.dyn_graph(hb)
        return dyn_ei, dyn_ew

    def _batch_graph(self, ha, B):
        N = self.n_agents
        if self.model != 'dynamic' or self.graph_alpha < 1.0:
            ei = self.grid_adj.to(ha.device)
            offsets = torch.arange(B, device=ha.device) * N
            ei_batch = (ei.unsqueeze(0) + offsets.view(B,1,1)).view(2,-1)
            return ei_batch, None
        else:
            h_graph = ha.view(B, N, -1)
            ei_list, ew_list = [], []
            for b in range(B):
                ei, ew = self.dyn_graph(h_graph[b])
                ei_list.append(ei + b * N)
                ew_list.append(ew)
            return torch.cat(ei_list, dim=1), torch.cat(ew_list)

    def forward(self, obs, use_graph=False):
        ha = F.relu(self.actor_encoder(obs))
        hc = F.relu(self.critic_encoder(obs))
        if use_graph:
            B = obs.shape[0] // self.n_agents
            ei, ew = self._batch_graph(ha, B)
            for gnn in self.gnns:
                ha = gnn(ha, ei, ew)
        mu  = self.actor_mu(ha) * 0.1
        std = self.actor_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        return mu, std, self.critic(hc).squeeze(-1)

    @torch.no_grad()
    def get_action(self, obs):
        mu, std, val = self.forward(obs, use_graph=True)
        d = Normal(mu, std); a = d.sample()
        return a.cpu().numpy(), d.log_prob(a).sum(-1), val

    def evaluate(self, obs, actions):
        ha = F.relu(self.actor_encoder(obs))
        if obs.shape[0] % self.n_agents == 0:
            B = obs.shape[0] // self.n_agents
            ei, ew = self._batch_graph(ha, B)
            for gnn in self.gnns:
                ha = gnn(ha, ei, ew)
        mu  = self.actor_mu(ha) * 0.1
        std = self.actor_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        d   = Normal(mu, std)
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1)

    def evaluate_critic(self, obs):
        hc = F.relu(self.critic_encoder(obs))
        return self.critic(hc).squeeze(-1)
