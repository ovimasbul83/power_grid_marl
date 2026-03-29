import numpy as np
import torch
import matplotlib.pyplot as plt

COLORS = {
    'baseline': '#FF9800', 'gcn': '#2196F3',
    'gat': '#4CAF50', 'dynamic': '#E91E63',
}
LABELS = {
    'baseline': 'Indep. PPO', 'gcn': 'GCN-MAPPO',
    'gat': 'GAT-MAPPO', 'dynamic': 'Dynamic-MAPPO (ours)',
}


def disturbance_test(controller, env, dist_step=100, dist_mag=0.2,
                     n_steps=500, is_policy=False, device='cpu'):
    obs, _ = env.reset()
    omega_h, pm_h = [], []
    for t in range(n_steps):
        if t == dist_step: env.load_dist[0] += dist_mag
        if is_policy:
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                acts, _, _ = controller.get_action(obs_t)
            actions = acts.squeeze(-1)
        else:
            actions = controller.act(obs)
        obs, _, _, trunc, _ = env.step(actions)
        omega_h.append(obs[:,0].copy()); pm_h.append(env.Pm.copy())
        if trunc: break
    omega = np.array(omega_h)
    post  = np.abs(omega[dist_step:]).mean(axis=1)
    nadir = post.max()
    settled  = np.where(post < 0.01)[0]
    settling = settled[0] * env.dt if len(settled) > 0 else float('inf')
    ss_err   = np.abs(omega[-50:]).mean()
    return omega, np.array(pm_h), nadir, settling, ss_err


def plot_learning_curves(trained, n_generators, topology, window=10, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for model in trained:
        _, rews, fds = trained[model]
        c = COLORS.get(model, 'black'); lbl = LABELS.get(model, model)
        for ax, data in zip(axes, [rews, fds]):
            ax.plot(data, alpha=0.2, color=c, lw=1)
            sm = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(data)), sm, color=c, label=lbl, lw=2)
    axes[0].set_title('Reward (higher = better)'); axes[0].set_ylabel('Mean episode reward')
    axes[1].set_title('Frequency deviation (lower = better)'); axes[1].set_ylabel('Mean |ω| (pu)')
    for ax in axes: ax.set_xlabel('Training update'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.suptitle(f'MARL Training — {n_generators} generators, {topology} topology', fontsize=13)
    plt.tight_layout()
    if show: plt.show()
    return fig


def plot_disturbance_response(all_res, dt=0.01, show=True):
    ctrl_c = {'Droop': '#9E9E9E', 'PI': '#607D8B', 'AGC': '#37474F'}
    marl_c = {'Indep. PPO': COLORS['baseline'], 'GCN-MAPPO': COLORS['gcn'],
               'GAT-MAPPO': COLORS['gat'], 'Dynamic-MAPPO': COLORS['dynamic']}
    all_c  = {**ctrl_c, **marl_c}
    t_ax   = np.arange(500) * dt
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for name, (omega, pm, *_) in all_res.items():
        c  = all_c.get(name, 'black')
        ls = '--' if name in ctrl_c else '-'
        lw = 1.5 if name in ctrl_c else 2.2
        axes[0].plot(t_ax[:len(omega)], omega.mean(axis=1), color=c, ls=ls, lw=lw, label=name)
        axes[1].plot(t_ax[:len(pm)],    pm.mean(axis=1),   color=c, ls=ls, lw=lw, label=name)
    for ax in axes:
        ax.axvline(x=100*dt, color='red', ls=':', alpha=0.8, label='Disturbance')
        ax.grid(alpha=0.3)
    axes[0].axhline(y= 0.01, color='gray', lw=0.8, ls=':', alpha=0.6)
    axes[0].axhline(y=-0.01, color='gray', lw=0.8, ls=':', alpha=0.6)
    axes[0].set_ylabel('Mean ω (pu)', fontsize=12)
    axes[0].set_title('Step disturbance response — +20% load at t=1s', fontsize=13)
    axes[0].legend(fontsize=9, ncol=2)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Mean Pm (pu)', fontsize=12)
    axes[1].set_title('Generator power output response', fontsize=13)
    plt.tight_layout()
    if show: plt.show()
    return fig


def print_metrics_table(all_res):
    print(f'\n{"Controller":<22} {"Nadir":>10} {"Settling(s)":>13} {"SS Error":>10}')
    print('-' * 58)
    for name, (_, _, nadir, settling, ss_err) in all_res.items():
        st  = f'{settling:.3f}' if settling != float('inf') else '    ∞'
        tag = ' ◄' if 'Dynamic' in name else ''
        print(f'{name:<22} {nadir:>10.5f} {st:>13} {ss_err:>10.5f}{tag}')
