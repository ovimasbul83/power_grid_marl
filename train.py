import argparse
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps',    type=int, default=200_000)
    p.add_argument('--n-gen',    type=int, default=4)
    p.add_argument('--topology', type=str, default='ring', choices=['ring', 'mesh', 'star'])
    p.add_argument('--hidden',   type=int, default=128)
    p.add_argument('--n-steps',  type=int, default=1024)
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--device',   type=str, default='auto')
    p.add_argument('--models',   nargs='+', default=['baseline', 'gcn', 'gat', 'dynamic'])
    p.add_argument('--save-dir', type=str, default='checkpoints')
    return p.parse_args()


def build_policy(model, env, hidden_dim):
    from power_grid_marl.policies import BaselinePolicy, GNNPolicy
    adj = env.get_adjacency(); n = env.n_generators
    if model == 'baseline':
        return BaselinePolicy(env.obs_dim, hidden_dim=hidden_dim)
    top_k = 2 if model == 'dynamic' else min(3, n-1)
    return GNNPolicy(env.obs_dim, n_agents=n, model=model,
                     hidden_dim=hidden_dim, grid_adj=adj, top_k=top_k)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    print(f'Device: {device} | Steps: {args.steps:,} | Topology: {args.topology}')

    from power_grid_marl.env import PowerGridEnv
    from power_grid_marl.trainer import train
    import os; os.makedirs(args.save_dir, exist_ok=True)

    for model in args.models:
        print(f'\n=== {model.upper()} ===')
        env = PowerGridEnv(n_generators=args.n_gen, topology=args.topology, seed=args.seed)
        policy = build_policy(model, env, args.hidden)
        policy, _, _ = train(env, policy, total_steps=args.steps,
                              n_steps=args.n_steps, device=device)
        ckpt = os.path.join(args.save_dir, f'{model}.pt')
        torch.save(policy.state_dict(), ckpt)
        print(f'  saved → {ckpt}')

    print('\nDone. Run evaluate.py to see results.')


if __name__ == '__main__':
    main()
