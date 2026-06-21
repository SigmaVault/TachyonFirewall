from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    L: float = 1.0
    gamma: float = 1.0
    damping: float = 0.35
    eps0: float = 0.02
    sign: float = +1.0
    t_total: float = 12.0
    fps: int = 30
    nx: int = 400

    @property
    def n_frames(self) -> int:
        return int(self.t_total * self.fps)

    @property
    def dt(self) -> float:
        return self.t_total / self.n_frames

    @property
    def x_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.L, self.nx)

    @property
    def sqrt2(self) -> float:
        return np.sqrt(2.0)

    @property
    def t_vac(self) -> float:
        return self.sign / self.sqrt2

    @property
    def m2_vac(self) -> float:
        return DDV(self.t_vac)


def V(T):
    return -T**2 + T**4


def DV(T):
    return -2.0 * T + 4.0 * T**3


def DDV(T):
    return -2.0 + 12.0 * T**2


def integrate_phi(params: Params) -> np.ndarray:
    n = params.n_frames
    dt = params.dt
    phi = np.empty(n + 1)
    vel = np.empty(n + 1)
    phi[0], vel[0] = params.eps0, 0.0
    s = params.sign
    for k in range(n):
        a = -DV(phi[k]) - params.damping * vel[k]
        if abs(phi[k]) < 0.05:
            a += s * 1e-3
        phi_half = phi[k] + 0.5 * vel[k] * dt
        a_half = -DV(phi_half) - params.damping * vel[k]
        vel[k + 1] = vel[k] + a_half * dt
        phi[k + 1] = phi_half + 0.5 * vel[k + 1] * dt
    return phi
