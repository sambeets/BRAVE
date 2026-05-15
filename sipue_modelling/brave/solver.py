from __future__ import annotations

import pulp

from brave.config import Config


def make_solver_lp(cfg: Config):
    kw = {"msg": cfg.solver_msg}
    if cfg.solver_time_limit_sec is not None:
        kw["timeLimit"] = cfg.solver_time_limit_sec

    name = (cfg.solver_lp or "").strip().lower()
    if name in ("highs", "highs_api", "highs_py"):
        if not hasattr(pulp, "HiGHS"):
            raise RuntimeError("PuLP HiGHS solver is not available in this environment.")
        return pulp.HiGHS(**kw)

    if name in ("highs_cmd", "highs-cli", "highscli"):
        if not hasattr(pulp, "HiGHS_CMD"):
            raise RuntimeError("PuLP HiGHS_CMD solver is not available in this environment.")
        return pulp.HiGHS_CMD(**kw)

    raise ValueError(f"Unknown cfg.solver_lp='{cfg.solver_lp}'.")
