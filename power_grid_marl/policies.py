import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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

    def _gnn_forward(self, h):
        B = h.shape[0] // self.n_agents
        h_graph = h.view(B, self.n_agents, -1); out = []
        for b in range(B):
            hb = h_graph[b]
            ei, ew = (self.dyn_graph(hb) if self.model == 'dynamic'
                      else (self.grid_adj.to(hb.device), None))
            for gnn in self.gnns: hb = gnn(hb, ei, ew)
            out.append(hb)
        return torch.stack(out).view(B * self.n_agents, -1)

    def forward(self, obs, use_graph=False):
        ha  = F.relu(self.actor_encoder(obs))
        hc  = F.relu(self.critic_encoder(obs))
        if use_graph: ha = self._gnn_forward(ha)
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
        if obs.shape[0] % self.n_agents == 0: ha = self._gnn_forward(ha)
        mu  = self.actor_mu(ha) * 0.1
        std = self.actor_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        d   = Normal(mu, std)
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1)

    def evaluate_critic(self, obs):
        hc = F.relu(self.critic_encoder(obs))
        if obs.shape[0] % self.n_agents == 0: hc = self._gnn_forward(hc)
        return self.critic(hc).squeeze(-1)
