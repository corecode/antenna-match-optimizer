import itertools
import warnings
from enum import Enum, auto
from typing import Iterator

import numpy as np
import skrf as rf
from numpy.typing import NDArray
from scipy.optimize import minimize

from . import passives
from .passives import ComponentList, Toleranced

ArchParams = tuple[float, float]


class Arch(Enum):
    LshuntCseries = auto()
    CshuntLseries = auto()
    LseriesCshunt = auto()
    CseriesLshunt = auto()


class OptimizeResult:
    def __init__(self, arch: Arch, x: ArchParams, ntwk: rf.Network):
        self.arch, self.x, self.ntwk = arch, x, ntwk


def matching_network(arch: Arch, x: ArchParams, ntwk: rf.Network) -> rf.Network:
    L = x[0] * 1e-9
    C = x[1] * 1e-12
    line = rf.DefinedGammaZ0(frequency=ntwk.frequency)

    def named(label: str, matching_ntwk: rf.Network) -> rf.Network:
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


def matching_objective(
    x: ArchParams, arch: Arch, ntwk: rf.Network, frequency: str | None
) -> float:
    matched = matching_network(arch, x, ntwk)
    if frequency:
        s_mag = matched[frequency].s_mag
    else:
        s_mag = matched.s_mag
    reflected_power = np.sum(s_mag**2.0)
    return float(reflected_power)


def optimize(ntwk: rf.Network, frequency: str | None = None) -> list[OptimizeResult]:
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


def closest_values(
    value: float, components: ComponentList
) -> list[tuple[float, float]]:
    rel = components[:, 0] / value - 1.0
    signs = np.sign(rel)
    best = np.argsort(np.abs(rel))
    result = [components[best[0]]]
    if np.abs(rel[best[0]]) > 1e-3:
        for i in range(1, len(rel)):
            result.append(components[best[i]])
            if signs[best[i]] != signs[best[0]]:
                break
    return result


def expand_tolerance(val_and_tolerance: Toleranced) -> list[float]:
    val = val_and_tolerance[0]
    tolerance = val_and_tolerance[1]
    if tolerance > 0.0:
        return [val, val - tolerance, val + tolerance]
    else:
        return [val]


def component_combinations(
    arch: Arch,
    x: ArchParams,
    inductors: ComponentList = passives.INDUCTORS,
    capacitors: ComponentList = passives.CAPACITORS,
) -> Iterator[tuple[tuple[Arch, ArchParams], ArchParams]]:
    l_comps = closest_values(x[0], inductors)
    c_comps = closest_values(x[1], capacitors)
    for l_comp, c_comp in itertools.product(l_comps, c_comps):
        l_tols = expand_tolerance(l_comp)
        c_tols = expand_tolerance(c_comp)
        for l_val, c_val in itertools.product(l_tols, c_tols):
            yield (arch, (l_comp[0], c_comp[0])), (l_val, c_val)
