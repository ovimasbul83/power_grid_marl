import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class PowerGridEnv(gym.Env):
    def __init__(self, n_generators=4, dt=0.01, max_steps=500,
                 topology='ring', disturbance_std=0.05, seed=None):
        super().__init__()
        self.n_generators = n_generators
        self.dt = dt
        self.max_steps = max_steps
        self.disturbance_std = disturbance_std
        self.M     = np.ones(n_generators) * 10.0
        self.D     = np.ones(n_generators) * 1.0
        self.P_max = np.ones(n_generators) * 1.0
        self.P_min = np.zeros(n_generators)
        self.P_nom = np.ones(n_generators) * 0.5
        self.T_gov = np.ones(n_generators) * 0.1
        self.B     = self._build_susceptance(topology, n_generators)
        self.obs_dim = 5
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(n_generators, self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(-0.1, 0.1, shape=(n_generators,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def _build_susceptance(self, topology, n):
        B = np.zeros((n, n)); b = 5.0
        if topology == 'ring':
            for i in range(n): j = (i+1)%n; B[i,j]=b; B[j,i]=b
        elif topology == 'mesh':
            for i in range(n):
                for j in range(i+1, n): B[i,j]=b; B[j,i]=b
        elif topology == 'star':
            for i in range(1, n): B[0,i]=b; B[i,0]=b
        for i in range(n): B[i,i] = -np.sum(B[i,:])
        return B

    def _compute_Pe(self):
        Pe = np.zeros(self.n_generators)
        for i in range(self.n_generators):
            for j in range(self.n_generators):
                if i != j and self.B[i,j] != 0:
                    Pe[i] += self.B[i,j] * (self.delta[i] - self.delta[j])
        return Pe

    def _get_obs(self):
        return np.stack([self.omega, self.delta, self.Pe,
                         self.Pm, self.load_dist], axis=1).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.omega     = self.rng.normal(0, 0.01, self.n_generators)
        self.delta     = self.rng.normal(0, 0.01, self.n_generators)
        self.Pm        = self.P_nom.copy()
        self.Pe        = self._compute_Pe()
        self.load_dist = self.rng.normal(0, self.disturbance_std, self.n_generators)
        self.step_count = 0
        self.disturbance_event = None
        if self.rng.random() < 0.5:
            t_event = self.rng.integers(50, 200)
            g_event = self.rng.integers(0, self.n_generators)
            mag = self.rng.uniform(0.1, 0.25)
            self.disturbance_event = (t_event, g_event, mag)
        return self._get_obs(), {}

    def step(self, actions):
        actions = np.clip(actions, -0.1, 0.1)
        Pm_ref  = np.clip(self.Pm + actions, self.P_min, self.P_max)
        self.Pm += (self.dt / self.T_gov) * (Pm_ref - self.Pm)
        self.load_dist += self.rng.normal(0, 0.005, self.n_generators)
        self.load_dist  = np.clip(self.load_dist, -0.5, 0.5)
        if self.disturbance_event is not None:
            t_event, g_event, mag = self.disturbance_event
            if self.step_count == t_event:
                self.load_dist[g_event] += mag
        self.Pe = self._compute_Pe()
        w0 = 2 * np.pi * 60
        dw = (1/self.M) * (self.Pm - self.Pe - self.D*self.omega - self.load_dist)
        self.omega = np.clip(self.omega + self.dt*dw, -2.0, 2.0)
        self.delta = np.clip(self.delta + self.dt*w0*self.omega, -np.pi, np.pi)
        rewards = (
            -1.0 * np.abs(self.omega)
            -0.5 * np.abs(np.mean(self.omega))
            -0.01 * np.abs(actions) / 0.1
            -0.1  * np.abs(self.Pm - self.P_nom)
        )
        if self.step_count + 1 >= self.max_steps:
            rewards -= 2.0 * np.mean(np.abs(self.omega))
        self.step_count += 1
        info = {'mean_freq_dev': np.mean(np.abs(self.omega)),
                'max_freq_dev':  np.max(np.abs(self.omega))}
        return self._get_obs(), rewards, False, self.step_count >= self.max_steps, info

    def get_adjacency(self):
        src, dst = [], []
        for i in range(self.n_generators):
            for j in range(self.n_generators):
                if i != j and self.B[i,j] != 0:
                    src.append(i); dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)
