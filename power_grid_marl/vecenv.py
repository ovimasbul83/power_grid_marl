import numpy as np
import torch


class VecPowerGridEnv:
    def __init__(self, n_envs, n_generators, topology, seed=None):
        self.n_envs    = n_envs
        self.n_agents  = n_generators
        self.obs_dim   = 5
        self.dt        = 0.01
        self.max_steps = 500
        self.disturbance_std = 0.05

        from .env import PowerGridEnv
        _env = PowerGridEnv(n_generators=n_generators, topology=topology, seed=0)
        self.M   = _env.M;   self.D   = _env.D
        self.P_max=_env.P_max; self.P_min=_env.P_min; self.P_nom=_env.P_nom
        self.T_gov=_env.T_gov; self.B_mat=_env.B
        self._adj = _env.get_adjacency()

        self.rng = np.random.default_rng(seed)
        self._init_states()

    def _init_states(self):
        E, N = self.n_envs, self.n_agents
        self.omega      = np.zeros((E, N))
        self.delta      = np.zeros((E, N))
        self.Pm         = np.zeros((E, N))
        self.Pe         = np.zeros((E, N))
        self.load_dist  = np.zeros((E, N))
        self.step_count = np.zeros(E, dtype=int)
        self.dist_t = np.full(E, -1, dtype=int)
        self.dist_g = np.zeros(E, dtype=int)
        self.dist_m = np.zeros(E)

    def _compute_Pe_batch(self):
        return -(self.delta @ self.B_mat.T)

    def reset(self):
        E, N = self.n_envs, self.n_agents
        self.omega     = self.rng.normal(0, 0.01, (E, N))
        self.delta     = self.rng.normal(0, 0.01, (E, N))
        self.Pm        = np.tile(self.P_nom, (E, 1))
        self.load_dist = self.rng.normal(0, self.disturbance_std, (E, N))
        self.Pe        = self._compute_Pe_batch()
        self.step_count[:] = 0
        has_event = self.rng.random(E) < 0.5
        self.dist_t = np.where(has_event, self.rng.integers(50, 200, E), -1)
        self.dist_g = self.rng.integers(0, N, E)
        self.dist_m = self.rng.uniform(0.1, 0.25, E)
        return self._get_obs()

    def _get_obs(self):
        return np.stack([self.omega, self.delta, self.Pe,
                         self.Pm, self.load_dist], axis=-1).astype(np.float32)

    def step(self, actions):
        E, N = self.n_envs, self.n_agents
        actions = np.clip(actions, -0.1, 0.1)
        Pm_ref  = np.clip(self.Pm + actions, self.P_min, self.P_max)
        self.Pm += (self.dt / self.T_gov) * (Pm_ref - self.Pm)

        self.load_dist += self.rng.normal(0, 0.005, (E, N))
        self.load_dist  = np.clip(self.load_dist, -0.5, 0.5)

        active = (self.dist_t >= 0) & (self.step_count == self.dist_t)
        for e in np.where(active)[0]:
            self.load_dist[e, self.dist_g[e]] += self.dist_m[e]
        self.dist_t[active] = -1

        self.Pe = self._compute_Pe_batch()
        w0 = 2 * np.pi * 60
        dw = (1/self.M) * (self.Pm - self.Pe - self.D*self.omega - self.load_dist)
        self.omega = np.clip(self.omega + self.dt*dw, -2.0, 2.0)
        self.delta = np.clip(self.delta + self.dt*w0*self.omega, -np.pi, np.pi)

        rewards = -np.abs(self.omega)/2.0 - 0.05*np.abs(actions)/0.1
        self.step_count += 1
        truncs = self.step_count >= self.max_steps

        if truncs.any():
            done_idx = np.where(truncs)[0]
            self.omega[done_idx]     = self.rng.normal(0, 0.01, (len(done_idx), N))
            self.delta[done_idx]     = self.rng.normal(0, 0.01, (len(done_idx), N))
            self.Pm[done_idx]        = self.P_nom
            self.load_dist[done_idx] = self.rng.normal(0, self.disturbance_std, (len(done_idx), N))
            self.Pe[done_idx]        = self._compute_Pe_batch()[done_idx]
            self.step_count[done_idx]= 0
            has_ev = self.rng.random(len(done_idx)) < 0.5
            self.dist_t[done_idx] = np.where(has_ev, self.rng.integers(50,200,len(done_idx)), -1)
            self.dist_g[done_idx] = self.rng.integers(0, N, len(done_idx))
            self.dist_m[done_idx] = self.rng.uniform(0.1, 0.25, len(done_idx))

        mean_fd = np.mean(np.abs(self.omega))
        return self._get_obs(), rewards, truncs, {'mean_freq_dev': mean_fd}

    def get_adjacency(self):
        return self._adj
