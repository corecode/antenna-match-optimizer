import warnings
from enum import Enum, auto

import numpy as np
import skrf as rf
from scipy.optimize import minimize

from . import passives


class Arch(Enum):
    LshuntCseries = auto()
    CshuntLseries = auto()
    LseriesCshunt = auto()
    CseriesLshunt = auto()


class OptimizeResult:
    def __init__(self, arch: Arch, x: list[float], ntwk: rf.Network):
        self.arch, self.x, self.ntwk = arch, x, ntwk


def matching_network(arch: Arch, x: np.ndarray, ntwk: rf.Network) -> rf.Network:
    L = x[0] * 1e-9
    C = x[1] * 1e-12
    line = rf.DefinedGammaZ0(frequency=ntwk.frequency)

    def named(label: str, matching_ntwk: rf.Network):
        n = matching_ntwk**ntwk
        n.name = f"{label}-{ntwk.name}"
        return n

    match arch:
        case Arch.LshuntCseries:
            return named(
                f"Lshunt{L}nH-C{C}pF", line.shunt_inductor(L) ** line.capacitor(C)
            )
        case Arch.CshuntLseries:
            return named(
                f"Cshunt{C}pF-C{L}nH", line.shunt_capacitor(C) ** line.inductor(L)
            )
        case Arch.LseriesCshunt:
            return named(
                f"L{L}nH-Cshunt{C}pF", line.inductor(L) ** line.shunt_capacitor(C)
            )
        case Arch.CseriesLshunt:
            return named(
                f"C{C}pF-Cshunt{L}nH", line.capacitor(C) ** line.shunt_inductor(L)
            )


def matching_objective(x, arch: Arch, ntwk: rf.Network, frequency: str | None) -> float:
    matched = matching_network(arch, x, ntwk)
    if frequency:
        s_mag = matched[frequency].s_mag
    else:
        s_mag = matched.s_mag
    reflected_power = np.sum(np.array(s_mag) ** 2.0)
    return float(reflected_power)


def optimize(ntwk: rf.Network, frequency: str | None = None):
    # start at geometric mean
    x0 = (
        (np.max(passives.INDUCTORS[:, 0]) * np.min(passives.INDUCTORS[:, 0])) ** 0.5,
        (np.max(passives.CAPACITORS[:, 0]) * np.min(passives.CAPACITORS[:, 0])) ** 0.5,
    )
    bounds = (
        (1e-3, 2 * np.max(passives.INDUCTORS[:, 0])),
        (1e-3, 2 * np.max(passives.CAPACITORS[:, 0])),
    )
    results = []
    for arch in Arch:
        # optimize sometimes warns if it runs over bounds
        with warnings.catch_warnings(action="ignore"):
            res = minimize(
                matching_objective,
                x0,
                args=(arch, ntwk, frequency),
                method="SLSQP",
                bounds=bounds,
            )
        matched_ntwk = matching_network(arch, res.x, ntwk)
        results.append(OptimizeResult(arch=arch, x=res.x, ntwk=matched_ntwk))
    return results
