from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

import numpy as np

from src.physics import Params, V, DDV, DV, integrate_phi

BG = "#0b0e1a"
FG = "#e8ecf5"
ACCENT = "#ff5a7a"
ACCENT2 = "#5ad1ff"
GRID = "#1c2336"


@dataclass
class Scene:
    fig: Any
    ax_str: Any
    ax_v: Any
    string_line: Any
    amp_marker: Any
    ball: Circle
    trail: Any
    trail_x: list[float]
    trail_y: list[float]
    stage_text: Any
    phi_text: Any


def stage_of(t: float) -> str:
    if t < 2.2:
        return ("Stage 1 — tachyon instability:  "
                r"$\varphi\sim\cosh(\Gamma t)$,  $m^2<0$")
    if t < 5.5:
        return ("Stage 2 — condensation:  rolling down to the minimum "
                r"$T_{\mathrm{vac}}=\pm1/\sqrt{2}$")
    return ("Stage 3 — spontaneous symmetry breaking:  new vacuum,  "
            r"$m^2=V''(T_{\mathrm{vac}})=+4>0$")


def build_scene(params: Params) -> Scene:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#88a",
        "axes.labelcolor": "#ccd",
        "xtick.color": "#aab",
        "ytick.color": "#aab",
        "text.color": "#dde",
    })

    sqrt2 = params.sqrt2
    X = params.x_grid
    L = params.L

    fig = plt.figure(figsize=(12.8, 6.4), facecolor=BG)
    fig.canvas.manager.set_window_title("TachyonFirewall")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22,
                          left=0.07, right=0.97, top=0.86, bottom=0.12)

    ax_str = fig.add_subplot(gs[0, 0], facecolor=BG)
    ax_str.set_xlim(0, L)
    ax_str.set_ylim(-1.35, 1.35)
    ax_str.set_xlabel("x  (along the open string)", color="#aab")
    ax_str.set_ylabel("T(x, t)", color="#aab")
    ax_str.set_title(r"$T(x,t)=\varphi(t)\,\sin(\pi x/L)$",
                     color=FG, pad=10)
    ax_str.grid(True, color=GRID, lw=0.6)
    for s in ax_str.spines.values():
        s.set_color("#334")
    for tv in (+1 / sqrt2, -1 / sqrt2):
        ax_str.axhline(tv, color="#2a6", lw=0.7, ls=":", alpha=0.5)
        ax_str.axhline(-tv, color="#2a6", lw=0.7, ls=":", alpha=0.5)
    ax_str.axhline(0, color="#445", lw=0.6)
    for xb in (0, L):
        ax_str.axvline(xb, color=ACCENT, lw=2.5, alpha=0.55)
    ax_str.text(0, -1.28, "D-brane", color=ACCENT, fontsize=8, ha="center")
    ax_str.text(L, -1.28, "D-brane", color=ACCENT, fontsize=8, ha="center")
    string_line, = ax_str.plot([], [], color=ACCENT2, lw=2.4)
    amp_marker, = ax_str.plot([], [], "o", color=ACCENT, ms=6)

    ax_v = fig.add_subplot(gs[0, 1], facecolor=BG)
    T_GRID = np.linspace(-1.05, 1.05, 400)
    ax_v.plot(T_GRID, V(T_GRID), color=ACCENT2, lw=2.0)
    ax_v.axhline(0, color="#445", lw=0.6)
    ax_v.set_xlim(-1.05, 1.05)
    ax_v.set_ylim(-0.32, 0.22)
    ax_v.set_xlabel("T", color="#aab")
    ax_v.set_ylabel("V(T) = -T² + T⁴", color="#aab")
    ax_v.set_title("Tachyon effective potential", color=FG, pad=10)
    ax_v.grid(True, color=GRID, lw=0.6)
    for s in ax_v.spines.values():
        s.set_color("#334")
    for tv in (+1 / sqrt2, -1 / sqrt2):
        ax_v.plot(tv, V(tv), "o", color="#2a6", ms=7, zorder=3)
    ax_v.annotate(r"$T_{\mathrm{vac}}=+1/\sqrt{2}$",
                  xy=(+1 / sqrt2, V(+1 / sqrt2)),
                  xytext=(0.62, -0.05), color="#7d7", fontsize=9)
    ax_v.annotate(r"$T_{\mathrm{vac}}=-1/\sqrt{2}$",
                  xy=(-1 / sqrt2, V(-1 / sqrt2)),
                  xytext=(-0.95, -0.05), color="#7d7", fontsize=9)
    ax_v.annotate("unstable\nvacuum  T=0\n" + r"$m^2=-2<0$",
                  xy=(0, 0), xytext=(-0.30, 0.12), color="#d77", fontsize=8.5,
                  arrowprops=dict(arrowstyle="->", color="#d77", lw=0.8))

    ball = Circle((params.eps0, V(params.eps0)), radius=0.035, color=ACCENT, zorder=5)
    ax_v.add_patch(ball)
    trail, = ax_v.plot([], [], color=ACCENT, lw=1.0, alpha=0.45)

    fig.suptitle("TachyonFirewall — tachyon condensation on a string",
                 color=FG, fontsize=13, y=0.965)
    stage_text = fig.text(0.5, 0.91, "", color=ACCENT, fontsize=11,
                          ha="center", va="center")
    phi_text = ax_v.text(0.02, 0.97, "", transform=ax_v.transAxes,
                         color=FG, fontsize=9, va="top",
                         bbox=dict(facecolor="#11162a", edgecolor="#334",
                                   boxstyle="round,pad=0.3"))

    return Scene(
        fig=fig, ax_str=ax_str, ax_v=ax_v,
        string_line=string_line, amp_marker=amp_marker,
        ball=ball, trail=trail, trail_x=[], trail_y=[],
        stage_text=stage_text, phi_text=phi_text,
    )


def make_init(scene: Scene):
    def init():
        scene.string_line.set_data([], [])
        scene.amp_marker.set_data([], [])
        scene.trail.set_data([], [])
        return scene.string_line, scene.amp_marker, scene.trail, scene.ball
    return init


def make_update(scene: Scene, params: Params, phi: np.ndarray):
    X = params.x_grid
    L = params.L
    dt = params.dt

    def update(frame: int):
        t = frame * dt
        p = phi[frame % len(phi)]
        T_field = p * np.sin(np.pi * X / L)

        scene.string_line.set_data(X, T_field)
        scene.amp_marker.set_data([0.5], [p])

        scene.ball.center = (p, V(p))
        scene.trail_x.append(p)
        scene.trail_y.append(V(p))
        scene.trail.set_data(scene.trail_x, scene.trail_y)

        stage_label = stage_of(t)
        scene.stage_text.set_text(stage_label)
        scene.phi_text.set_text(
            f"t = {t:5.2f}    φ = {p:+.3f}    "
            f"V(φ) = {V(p):+.3f}    "
            f"V''(φ) = {(-2 + 12*p**2):+.2f}"
        )

        return (scene.string_line, scene.amp_marker, scene.trail, scene.ball,
                scene.stage_text, scene.phi_text)
    return update


def frame_stream():
    i = 0
    while True:
        yield i
        i += 1


def run(params: Params) -> None:
    phi = integrate_phi(params)
    scene = build_scene(params)
    init = make_init(scene)
    update = make_update(scene, params, phi)
    ani = FuncAnimation(
        scene.fig, update, frames=frame_stream, init_func=init,
        interval=1000 / params.fps, blit=False, repeat=True,
        cache_frame_data=False, save_count=params.n_frames,
    )
    print("TachyonFirewall: window open. Close the window to exit.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
