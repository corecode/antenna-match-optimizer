import io
import itertools
import os

import antenna_match_optimizer as mopt
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
    return buf.getvalue().decode("utf-8")


def plot_smith(ntwk: rf.Network) -> None:
    plt.figure(figsize=(3.5, 2.5), layout="constrained")
    ntwk.plot_s_smith()


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
    best = mopt.best_config(args, results)

    plot_smith(base)
    base_smith = {"svg": plot_to_svg(), "name": base.name}
    plot_vswr(base, frequency=frequency)
    base_vswr = {"svg": plot_to_svg(), "name": base.name}

    plot_smith(best.ntwk)
    plt.gca().get_legend().remove()
    best_smith = {"svg": plot_to_svg(), "name": best.ntwk[0].name}
    plt.figure(figsize=(3.5, 2.5), layout="constrained")
    plot_with_tolerance(best.ntwk[frequency])
    plt.gca().set_ylim(bottom=1.0)
    best_vswr = {"svg": plot_to_svg(), "name": best.ntwk[0].name}

    results_vswr = plot_architectures(results, frequency, func="s_vswr")

    return render_template(
        "optimize.html",
        base_smith=base_smith,
        base_vswr=base_vswr,
        best_smith=best_smith,
        best_vswr=best_vswr,
        results_vswr=results_vswr,
    )


def plot_architectures(
    results: list[mopt.OptimizeResult], frequency: str | None, func: str
):
    plots = []
    top_bound = np.max(
        [getattr(r.ntwk[frequency], f"max_{func}").s_re for r in results]
    )

    for arch, arch_results in itertools.groupby(results, lambda r: r.arch):
        plt.figure(figsize=(5, 3.5), layout="constrained")
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
