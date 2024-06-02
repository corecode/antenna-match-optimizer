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


def plot_smith(ntwk: rf.Network) -> dict[str, str]:
    plt.figure(figsize=(4, 4), layout="constrained")
    ntwk.plot_s_smith()
    return {"svg": plot_to_svg(), "name": ntwk.name}


def plot_vswr(ntwk: rf.Network, frequency: str | None) -> dict[str, str]:
    plt.figure(figsize=(6, 4), layout="constrained")
    ntwk.frequency.unit = "GHz"
    ntwk[frequency].plot_s_vswr()
    ax = plt.gca()
    ax.set_ylim(bottom=1.0)
    return {"svg": plot_to_svg(), "name": ntwk.name}


def plot_with_tolerance(ntws: rf.NetworkSet, func: str = "s_vswr", **kwargs):
    plotting_method = getattr(ntws[0], f"plot_{func}")
    plotting_method(**kwargs)
    ax = plt.gca()
    ax.fill_between(
        ntws[0].frequency.f,
        getattr(ntws, f"min_{func}").s_re[:, 0, 0],
        getattr(ntws, f"max_{func}").s_re[:, 0, 0],
        alpha=0.3,
        color=ax.get_lines()[-1].get_color(),
    )


# def plot_architectures(
#     variations: list[mopt.OptimizeResult],
#     frequency: str | None = None,
#     func: str = "s_vswr",
# ) -> dict[str, str]:
#     plt.figure(figsize=(12, 8), layout="constrained")
#     plt.subplots(
#         2,
#         2,
#         figsize=(10, 6),
#         sharex=True,
#         sharey=True,
#         gridspec_kw={"wspace": 0.02, "hspace": 0.02},
#     )

#     grouped_by_arch = itertools.groupby(variations, lambda v: v.arch)
#     for i, (_, arch_variations) in enumerate(grouped_by_arch):
#         ax = plt.subplot(2, 2, i + 1)
#         # plt.xticks(rotation=45)
#         if i < 2:
#             ax.set_xlabel(None)
#         for ntws in arch_variations:
#             plot_with_tolerance(ntws[frequency], func=func)

#     return {"svg": plot_to_svg()}


@bp.route("/optimize")
def optimize():
    matplotlib.style.use(os.path.join(os.path.dirname(__file__), "match.mplstyle"))

    base = make_detuned_antenna()
    frequency = "2.4-2.5GHz"

    ideal = mopt.optimize(base, frequency)
    results = mopt.evaluate_components(base, *ideal, frequency=frequency)

    base_smith = plot_smith(base)
    base_vswr = plot_vswr(base, frequency=frequency)

    results_vswr = plot_architectures(results, frequency, func="s_vswr")

    return render_template(
        "optimize.html",
        base_smith=base_smith,
        base_vswr=base_vswr,
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
        plt.figure(figsize=(6, 4), layout="constrained")
        for combination in arch_results:
            plot_with_tolerance(combination.ntwk[frequency], func=func)
            ax = plt.gca()
            ax.set_ylim(bottom=1.0, top=top_bound)
        plots.append({"svg": plot_to_svg(), "name": str(arch)})
    return plots
