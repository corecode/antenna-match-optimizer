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


class OptimizerArgs:
    def __init__(self, ntwk: rf.Network, frequency: str | None = None):
        self.ntwk = ntwk
        self.frequency = frequency
        if self.frequency:
            self.bandlimited_ntwk = self.ntwk[self.frequency]
        else:
            self.bandlimited_ntwk = self.ntwk


class OptimizeResult:
    def __init__(self, arch: Arch, x: ArchParams, ntwk: rf.Network | rf.NetworkSet):
        self.arch, self.x, self.ntwk = arch, x, ntwk


def matching_network(arch: Arch, x: ArchParams, ntwk: rf.Network) -> rf.Network:
    L = x[0] * 1e-9
    C = x[1] * 1e-12
    Lstr = f"{x[0]:#.3g}nH"
    Cstr = f"{x[1]:#.3g}pF"
    line = rf.DefinedGammaZ0(frequency=ntwk.frequency)

    def named(label: str, matching_ntwk: rf.Network) -> rf.Network:
        n = matching_ntwk**ntwk
        n.name = label
        return n

    match arch:
        case Arch.LshuntCseries:
            return named(
                f"{Lstr}|{Cstr}―",
                line.shunt_inductor(L) ** line.capacitor(C),
            )
        case Arch.CshuntLseries:
            return named(
                f"{Cstr}|{Lstr}―",
                line.shunt_capacitor(C) ** line.inductor(L),
            )
        case Arch.LseriesCshunt:
            return named(
                f"{Lstr}―{Cstr}|",
                line.inductor(L) ** line.shunt_capacitor(C),
            )
        case Arch.CseriesLshunt:
            return named(
                f"{Cstr}―{Lstr}|",
                line.capacitor(C) ** line.shunt_inductor(L),
            )


def matching_objective(x: ArchParams, arch: Arch, args: OptimizerArgs) -> float:
    matched = matching_network(arch, x, args.bandlimited_ntwk)
    return float(np.sum(matched.s_mag**2.0))


def optimize(args: OptimizerArgs) -> list[OptimizeResult]:
    # start at geometric mean
    x0 = (
        (np.max(passives.INDUCTORS[:, 0]) * np.min(passives.INDUCTORS[:, 0])) ** 0.5,
        (np.max(passives.CAPACITORS[:, 0]) * np.min(passives.CAPACITORS[:, 0])) ** 0.5,
    )
    bounds = (
        (1e-3, 2 * np.max(passives.INDUCTORS[:, 0])),
        (1e-3, 2 * np.max(passives.CAPACITORS[:, 0])),
    )

    def optimize_arch(arch: Arch) -> OptimizeResult:
        # optimize sometimes warns if it runs over bounds
        with warnings.catch_warnings(action="ignore"):
            res = minimize(
                matching_objective,
                x0,
                args=(arch, args),
                method="SLSQP",
                bounds=bounds,
            )
        matched_ntwk = matching_network(arch, res.x, args.bandlimited_ntwk)
        return OptimizeResult(arch=arch, x=res.x, ntwk=matched_ntwk)

    results = [optimize_arch(a) for a in Arch]

    return results


def closest_values(
    value: float, components: ComponentList
) -> list[tuple[float, float]]:
    if np.isclose(value, 0.001) or np.isclose(value, np.max(components[:, 0]) * 2):
        return [(value, 0.0)]
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


Tag = tuple[Arch, ArchParams]


def component_combinations(
    arch: Arch,
    x: ArchParams,
    inductors: ComponentList = passives.INDUCTORS,
    capacitors: ComponentList = passives.CAPACITORS,
) -> Iterator[tuple[Tag, ArchParams]]:
    l_comps = closest_values(x[0], inductors)
    c_comps = closest_values(x[1], capacitors)
    for l_comp, c_comp in itertools.product(l_comps, c_comps):
        l_tols = expand_tolerance(l_comp)
        c_tols = expand_tolerance(c_comp)
        for l_val, c_val in itertools.product(l_tols, c_tols):
            yield (arch, (l_comp[0], c_comp[0])), (l_val, c_val)


def evaluate_components(
    args: OptimizerArgs,
    *minima: OptimizeResult,
    inductors: ComponentList = passives.INDUCTORS,
    capacitors: ComponentList = passives.CAPACITORS,
) -> list[OptimizeResult]:
    tasks = []
    for minimum in minima:
        tasks += list(
            component_combinations(
                minimum.arch, minimum.x, inductors=inductors, capacitors=capacitors
            )
        )

    def make_tagged_network(
        tagged_values: tuple[Tag, ArchParams],
    ) -> tuple[Tag, rf.Network]:
        tag, values = tagged_values
        matched_ntwk = matching_network(tag[0], values, args.bandlimited_ntwk)
        return (tag, matched_ntwk)

    matched_ntwks = [make_tagged_network(t) for t in tasks]

    results = []
    for tag, ntwks in itertools.groupby(matched_ntwks, lambda n: n[0]):
        ntwk_set = rf.NetworkSet([n for _, n in ntwks])
        results.append(OptimizeResult(tag[0], x=tag[1], ntwk=ntwk_set))
    return results


def best_config(args: OptimizerArgs, configs: list[OptimizeResult]) -> OptimizeResult:
    scores = [np.sum(r.ntwk.max_s_mag.s_mag**2) for r in configs]
    best = np.argmin(scores)
    return configs[best]
