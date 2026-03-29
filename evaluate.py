import argparse
import warnings
import numpy as np
import torch
import os

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints-dir', type=str, default='checkpoints')
    p.add_argument('--n-gen',    type=int, default=4)
    p.add_argument('--topology', type=str, default='ring', choices=['ring', 'mesh', 'star'])
    p.add_argument('--hidden',   type=int, default=128)
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--device',   type=str, default='auto')
    p.add_argument('--save-fig', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device

    from power_grid_marl.env import PowerGridEnv
    from power_grid_marl.controllers import DroopController, PIController, AGCController
    from power_grid_marl.policies import BaselinePolicy, GNNPolicy
    from power_grid_marl.utils import disturbance_test, plot_disturbance_response, print_metrics_table

    n   = args.n_gen
    env = PowerGridEnv(n_generators=n, topology=args.topology, seed=args.seed)
    adj = env.get_adjacency()
    all_res = {}

    for name, ctrl in [('Droop', DroopController(n)),
                        ('PI',    PIController(n)),
                        ('AGC',   AGCController(n))]:
        ctrl.reset()
        all_res[name] = disturbance_test(ctrl, env)

    policy_map = {
        'baseline': ('Indep. PPO',    BaselinePolicy(env.obs_dim, hidden_dim=args.hidden)),
        'gcn':      ('GCN-MAPPO',     GNNPolicy(env.obs_dim, n_agents=n, model='gcn',
                                                hidden_dim=args.hidden, grid_adj=adj)),
        'gat':      ('GAT-MAPPO',     GNNPolicy(env.obs_dim, n_agents=n, model='gat',
                                                hidden_dim=args.hidden, grid_adj=adj)),
        'dynamic':  ('Dynamic-MAPPO', GNNPolicy(env.obs_dim, n_agents=n, model='dynamic',
                                                hidden_dim=args.hidden, grid_adj=adj, top_k=2)),
    }
    for key, (display, policy) in policy_map.items():
        ckpt = os.path.join(args.checkpoints_dir, f'{key}.pt')
        if not os.path.exists(ckpt):
            print(f'[skip] {display} — {ckpt} not found')
            continue
        policy.load_state_dict(torch.load(ckpt, map_location=device))
        policy.eval()
        all_res[display] = disturbance_test(policy, env, is_policy=True, device=device)
        print(f'[ok] {display}')

    print_metrics_table(all_res)
    fig = plot_disturbance_response(all_res, dt=env.dt, show=args.save_fig is None)
    if args.save_fig:
        fig.savefig(args.save_fig, dpi=150, bbox_inches='tight')
        print(f'saved → {args.save_fig}')


if __name__ == '__main__':
    main()
