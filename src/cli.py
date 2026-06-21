from __future__ import annotations

import argparse

from src.physics import Params
from src.animation import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tachyon-firewall",
        description="TachyonFirewall — tachyon condensation on an open string.",
    )
    p.add_argument("--string-length", type=float, default=Params.L,
                   help="String length L (default: %(default)s)")
    p.add_argument("--gamma", type=float, default=Params.gamma,
                   help="Tachyon instability rate Gamma (default: %(default)s)")
    p.add_argument("--damping", type=float, default=Params.damping,
                   help="Damping coefficient (default: %(default)s)")
    p.add_argument("--eps0", type=float, default=Params.eps0,
                   help="Initial perturbation amplitude (default: %(default)s)")
    p.add_argument("--sign", type=float, default=Params.sign,
                   choices=(+1.0, -1.0),
                   help="Which vacuum to roll into, +1 or -1 (default: %(default)s)")
    p.add_argument("--t-total", type=float, default=Params.t_total,
                   help="Total physical time to simulate (default: %(default)s)")
    p.add_argument("--fps", type=int, default=Params.fps,
                   help="Animation frame rate (default: %(default)s)")
    p.add_argument("--nx", type=int, default=Params.nx,
                   help="Number of spatial grid points (default: %(default)s)")
    return p


def params_from_args(args: argparse.Namespace) -> Params:
    return Params(
        L=args.string_length,
        gamma=args.gamma,
        damping=args.damping,
        eps0=args.eps0,
        sign=args.sign,
        t_total=args.t_total,
        fps=args.fps,
        nx=args.nx,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    params = params_from_args(args)
    run(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())