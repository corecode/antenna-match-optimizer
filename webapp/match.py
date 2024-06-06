import io
import itertools
import math
import os
import re

import antenna_match_optimizer as mopt
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import schemdraw
import schemdraw.elements as elm
import skrf as rf
from flask import (
    Blueprint,
    render_template,
)

bp = Blueprint("match", __name__, url_prefix="/")


def make_detuned_antenna() -> rf.Network:
    ant = rf.Network("tests/2450AT18A100.s1p")
    line = rf.DefinedGammaZ0(frequency=ant.frequency)
    # random and unscientific perturbation
    ant_detune = (
        line.shunt_capacitor(0.5e-12)
        ** line.inductor(0.1e-9)
        ** line.shunt_capacitor(0.5e-12)
        ** ant
    )
    ant_detune.name = f"Detuned {ant.name}"
    return ant_detune


def plot_to_svg() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="svg")
    str = buf.getvalue().decode("utf-8")
    str = re.sub(r'(<svg [^>]*?) width="[^"]+" height="[^"]+"', r"\1", str)
    return str


def plot_smith(ntwk: rf.Network, frequency: str) -> None:
    plt.figure(figsize=(3.5, 2.5), layout="constrained")
    ntwk.plot_s_smith(label=None)
    plt.gca().set_prop_cycle(matplotlib.rcParams["axes.prop_cycle"])
    ntwk[frequency].plot_s_smith(linewidth=3)


def plot_vswr(ntwk: rf.Network, frequency: str | None) -> None:
    plt.figure(figsize=(3.5, 2.5), layout="constrained")
    ntwk.frequency.unit = "GHz"
    ntwk[frequency].plot_s_vswr()
    ax = plt.gca()
    ax.set_ylim(bottom=1.0)


def plot_with_tolerance(ntws: rf.NetworkSet, func: str = "s_vswr", **kwargs) -> None:
    plotting_method = getattr(ntws[0], f"plot_{func}")
    ntws[0].frequency.unit = "GHz"
    plotting_method(**kwargs)
    ax = plt.gca()
    ax.fill_between(
        ntws[0].frequency.f,
        getattr(ntws, f"min_{func}").s_re[:, 0, 0],
        getattr(ntws, f"max_{func}").s_re[:, 0, 0],
        alpha=0.3,
        color=ax.get_lines()[-1].get_color(),
    )


@bp.route("/optimize")
def optimize():
    matplotlib.style.use(os.path.join(os.path.dirname(__file__), "match.mplstyle"))

    base = make_detuned_antenna()
    frequency = "2.4-2.5GHz"
    frequency = "2.401-2.481GHz"

    args = mopt.OptimizerArgs(ntwk=base, frequency=frequency)
    ideal = mopt.optimize(args)
    results = mopt.evaluate_components(args, *ideal)
    best = mopt.expand_result(args, results[0])

    plot_smith(base, frequency=frequency)
    base_smith = plot_to_svg()
    plot_vswr(base, frequency=frequency)
    worst_vswr = plt.gca().get_ylim()[1]
    base_vswr = plot_to_svg()

    plot_smith(best.ntwk, frequency=frequency)
    plt.gca().get_legend().remove()
    best_smith = plot_to_svg()
    plt.figure(figsize=(3.5, 2.5), layout="constrained")
    plot_with_tolerance(best.ntwk[frequency])
    plt.gca().set_ylim(bottom=1.0, top=worst_vswr)
    best_vswr = plot_to_svg()

    best_schema = plot_schematic(best, antenna_name=base.name)

    results_vswr = plot_architectures(
        sorted(results, key=lambda r: r.arch.value),
        frequency,
        func="s_vswr",
        arch_limit=3,
    )

    return render_template(
        "optimize.html",
        base_name=base.name,
        base_smith=base_smith,
        base_vswr=base_vswr,
        best_name=best.ntwk[0].name,
        best_smith=best_smith,
        best_vswr=best_vswr,
        best_schema=best_schema,
        results_vswr=results_vswr,
    )


def plot_architectures(
    results: list[mopt.OptimizeResult],
    frequency: str | None,
    func: str,
    arch_limit: int,
):
    plots = []
    top_bound = np.max(
        [getattr(r.ntwk[frequency], f"max_{func}").s_re for r in results]
    )

    for arch, arch_results in itertools.groupby(results, lambda r: r.arch):
        if arch_limit > 0:
            arch_results = itertools.islice(arch_results, arch_limit)
        plt.figure(figsize=(3.5, 2.5), layout="constrained")
        for combination in arch_results:
            plot_with_tolerance(combination.ntwk[frequency], func=func)
            ax = plt.gca()
            ax.set_ylim(bottom=1.0, top=top_bound)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.00),
            ncol=1,
            fancybox=True,
        )
        plots.append({"svg": plot_to_svg(), "name": str(arch)})
    return plots


def plot_schematic(config: mopt.OptimizeResult, antenna_name: str = "") -> str:
    def pretty_value(value):
        if value < 10.0:
            val_str = f"{value:.1f}"
        else:
            val_str = f"{value:.0f}"

        if not math.isclose(value, float(val_str)):
            val_str = f"{value:#.3g}"

        return val_str

    text_offsets = {False: (0, 0.2), True: (-0.1, -0.1)}

    def make_ind(vertical=False):
        return elm.Inductor2(loops=2).label(
            f"{pretty_value(config.x[0])}nH", ofst=text_offsets[vertical]
        )

    def make_cap(vertical=False):
        return elm.Capacitor().label(
            f"{pretty_value(config.x[1])}pF", ofst=text_offsets[vertical]
        )

    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2)

        elm.Tag().left()

        match config.arch:
            case mopt.Arch.CseriesLshunt:
                make_cap()
            case mopt.Arch.LseriesCshunt:
                make_ind()
            case _:
                elm.Line()

        elm.Dot()
        d.push()

        match config.arch:
            case mopt.Arch.LshuntCseries | mopt.Arch.CseriesLshunt:
                make_ind(True).down()
            case mopt.Arch.CshuntLseries | mopt.Arch.LseriesCshunt:
                make_cap(True).down()

        elm.Ground()
        d.pop()

        match config.arch:
            case mopt.Arch.LshuntCseries:
                make_cap()
            case mopt.Arch.CshuntLseries:
                make_ind()
            case _:
                elm.Line()

        elm.Antenna().label(antenna_name, ofst=text_offsets[False])

    svg_str: str
    svg_str = d.get_imagedata("svg").decode("utf-8")
    svg_str = re.sub('"sans"', '"sans-serif"', svg_str)
    svg_str = re.sub(r'(<svg [^>]*?) height="[^"]+" width="[^"]+"', r"\1", svg_str)

    return svg_str
