from __future__ import annotations

import pulp

from brave.config import Config


def _solver_kwargs(cfg: Config):
    kw = {"msg": cfg.solver_msg}
    if cfg.solver_time_limit_sec is not None:
        kw["timeLimit"] = cfg.solver_time_limit_sec
    return kw


def _make_solver(name: str, cfg: Config):
    kw = _solver_kwargs(cfg)

    name = (name or "").strip().lower()
    if name in ("highs", "highs_api", "highs_py"):
        if not hasattr(pulp, "HiGHS"):
            raise RuntimeError("PuLP HiGHS solver is not available in this environment.")
        return pulp.HiGHS(**kw)

    if name in ("highs_cmd", "highs-cli", "highscli"):
        if not hasattr(pulp, "HiGHS_CMD"):
            raise RuntimeError("PuLP HiGHS_CMD solver is not available in this environment.")
        return pulp.HiGHS_CMD(**kw)

    if name in ("cbc", "pulp_cbc", "pulp_cbc_cmd"):
        solver = pulp.PULP_CBC_CMD(**kw)
        if not solver.available():
            raise RuntimeError("PuLP CBC solver is not available in this environment.")
        return solver

    raise ValueError(f"Unknown LP solver '{name}'.")


def make_solver_lp(cfg: Config):
    return _make_solver(cfg.solver_lp, cfg)
