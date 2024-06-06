import io
import itertools
import os
import re

import antenna_match_optimizer as mopt
import antenna_match_optimizer.plotting as mplt
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
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
            mopt.plotting.plot_with_tolerance(combination.ntwk[frequency], func=func)
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


def plot_schematic(ntwk: mopt.OptimizeResult, name: str):
    schema = mplt.plot_schematic(ntwk, antenna_name=name)

    svg_str: str
    svg_str = schema.get_imagedata("svg").decode("utf-8")
    svg_str = re.sub('"sans"', '"sans-serif"', svg_str)
    svg_str = re.sub(r'(<svg [^>]*?) height="[^"]+" width="[^"]+"', r"\1", svg_str)

    return svg_str


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

    mplt.plot_smith(base, frequency=frequency)
    base_smith = plot_to_svg()
    mplt.plot_vswr(base, frequency=frequency)
    worst_vswr = plt.gca().get_ylim()[1]
    base_vswr = plot_to_svg()

    mplt.plot_smith(best.ntwk, frequency=frequency)
    plt.gca().get_legend().remove()
    best_smith = plot_to_svg()
    plt.figure(figsize=(3.5, 2.5), layout="constrained")
    mplt.plot_with_tolerance(best.ntwk[frequency])
    plt.gca().set_ylim(bottom=1.0, top=worst_vswr)
    best_vswr = plot_to_svg()

    best_schema = plot_schematic(best, base.name)

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
