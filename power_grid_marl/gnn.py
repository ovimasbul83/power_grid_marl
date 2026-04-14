import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_weights=None):
        src, dst = edge_index[0], edge_index[1]
        msgs = x[src]
        if edge_weights is not None:
            msgs = msgs * edge_weights.unsqueeze(-1)
        agg = torch.zeros_like(x).index_add_(0, dst, msgs)
        cnt = torch.zeros(x.shape[0], 1, device=x.device).index_add_(
            0, dst, torch.ones(src.shape[0], 1, device=x.device))
        agg = agg / cnt.clamp(min=1.0)
        return F.relu(self.norm(self.linear(x + agg)))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        self.nh = n_heads; self.hd = out_dim // n_heads
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2*self.hd, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)
        self.norm  = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_weights=None):
        N = x.shape[0]
        Wh = self.W(x).view(N, self.nh, self.hd)
        src, dst = edge_index[0], edge_index[1]
        e = self.lrelu(self.attn(torch.cat([Wh[src], Wh[dst]], dim=-1))).squeeze(-1)
        exp_e = torch.exp(e - e.max())
        denom = torch.zeros(N, self.nh, device=x.device)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-8)
        if edge_weights is not None:
            alpha = alpha * edge_weights.unsqueeze(-1)
        msgs = (alpha.unsqueeze(-1) * Wh[src])
        out = torch.zeros(N, self.nh, self.hd, device=x.device)
        out.index_add_(0, dst, msgs)
        return F.relu(self.norm(out.view(N, -1)))


class DynamicGraphBuilder(nn.Module):
    def __init__(self, hidden_dim=128, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 1))

    def forward(self, h):
        n = h.shape[0]
        hi = h.unsqueeze(1).expand(-1, n, -1)
        hj = h.unsqueeze(0).expand(n, -1, -1)
        scores  = self.attn(torch.cat([hi, hj], dim=-1)).squeeze(-1)
        scores  = scores.masked_fill(torch.eye(n, device=h.device).bool(), float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        k = min(self.top_k, n-1)
        tv, ti = torch.topk(weights, k, dim=-1)
        src = torch.arange(n, device=h.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = ti.reshape(-1)
        ew = tv.reshape(-1)
        return torch.stack([src, dst], dim=0), ew
