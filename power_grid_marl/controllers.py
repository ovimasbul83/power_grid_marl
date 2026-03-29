import numpy as np


class DroopController:
    def __init__(self, n=4, R=0.05, clip=0.1):
        self.R = R; self.clip = clip

    def act(self, obs):
        return np.clip(-(1/self.R) * obs[:,0], -self.clip, self.clip)

    def reset(self): pass


class PIController:
    def __init__(self, n=4, Kp=10., Ki=2., dt=0.01, clip=0.1, windup=2.):
        self.Kp = Kp; self.Ki = Ki; self.dt = dt
        self.clip = clip; self.windup = windup
        self.n = n; self.integral = np.zeros(n)

    def act(self, obs):
        w = obs[:,0]
        self.integral = np.clip(self.integral + w*self.dt, -self.windup, self.windup)
        return np.clip(-self.Kp*w - self.Ki*self.integral, -self.clip, self.clip)

    def reset(self): self.integral = np.zeros(self.n)


class AGCController:
    def __init__(self, n=4, B=20., Ki=0.3, dt=0.01, clip=0.1):
        self.B = B; self.Ki = Ki; self.dt = dt; self.clip = clip
        self.alpha = np.ones(n)/n; self.integral = 0.

    def act(self, obs):
        ACE = self.B * np.mean(obs[:,0])
        self.integral += ACE * self.dt
        return np.clip(self.alpha * (-self.Ki*self.integral), -self.clip, self.clip)

    def reset(self): self.integral = 0.
