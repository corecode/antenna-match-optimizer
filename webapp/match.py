import io
import itertools
import re
from http import HTTPStatus
from pathlib import PurePath

import antenna_match_optimizer as mopt
import antenna_match_optimizer.plotting as mplt
import matplotlib.pyplot as plt
import numpy as np
import schemdraw
import schemdraw.elements as elm
import skrf as rf
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from matplotlib.figure import Figure

bp = Blueprint("match", __name__, url_prefix="/")


@bp.route("/")
def index():
    return redirect(url_for(".upload"), code=HTTPStatus.PERMANENT_REDIRECT)


@bp.route("/optimize", methods=["GET"])
def upload():
    pi_network = save_schematic(plot_pi_schematic())
    return render_template(
        "upload.html",
        pi_network=pi_network,
    )


@bp.route("/optimize", methods=["POST"])
def optimize():
    try:
        touchstone = request.files["touchstone"]
    except Exception:
        flash("No Touchstone file uploaded")
        return redirect(request.url, code=HTTPStatus.SEE_OTHER)

    touchstone_name = PurePath(touchstone.filename or "Noname")
    touchstone_data = touchstone.read().decode("utf-8")
    touchstone_io = io.StringIO(touchstone_data)
    touchstone_io.name = touchstone_name.name

    try:
        base = rf.Network(
            file=touchstone_io,
            name=touchstone_name.stem,
            f_unit="GHz",
        )
    except Exception:
        flash("Could not parse Touchstone file")
        return redirect(request.url, code=HTTPStatus.SEE_OTHER)

    frequency = request.form.get("frequency")
    if frequency is None or frequency == "":
        flash("You need to specify a frequency range")
        return redirect(request.url, code=HTTPStatus.SEE_OTHER)

    try:
        args = mopt.OptimizerArgs(ntwk=base, frequency=frequency)
    except Exception:
        flash("Frequency range is invalid")
        return redirect(request.url, code=HTTPStatus.SEE_OTHER)

    ideal = mopt.optimize(args)
    results = mopt.evaluate_components(args, *ideal)
    best = mopt.expand_result(args, results[0])

    base_smith_fig = mplt.plot_smith(base, frequency=frequency)
    base_smith = plot_to_svg(base_smith_fig)

    base_vswr_fig = mplt.plot_vswr(base, frequency=frequency)
    worst_vswr = base_vswr_fig.gca().get_ylim()[1]
    base_vswr = plot_to_svg(base_vswr_fig)

    best_smith_fig = mplt.plot_smith(best.ntwk, frequency=frequency)
    best_smith_fig.gca().get_legend().remove()
    best_smith = plot_to_svg(best_smith_fig)

    best_vswr_fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    mplt.plot_with_tolerance(best.ntwk[frequency], ax=ax)
    ax.set_ylim(bottom=1.0, top=worst_vswr)
    best_vswr = plot_to_svg(best_vswr_fig)

    best_schema = save_schematic(mplt.plot_schematic(best, antenna_name=base.name))

    results_vswr = plot_architectures(
        sorted(results, key=lambda r: r.arch.value),
        frequency,
        func="s_vswr",
        arch_limit=3,
    )

    plt.close("all")

    return render_template(
        "optimize.html",
        base_name=base.name,
        frequency=frequency,
        base_smith=base_smith,
        base_vswr=base_vswr,
        best_name=best.ntwk[0].name,
        best_smith=best_smith,
        best_vswr=best_vswr,
        best_schema=best_schema,
        results_vswr=results_vswr,
    )


def plot_to_svg(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    str = buf.getvalue().decode("utf-8")
    str = re.sub(r'(<svg [^>]*?) width="[^"]+" height="[^"]+"', r"\1", str)
    return str


def plot_architectures(
    results: list[mopt.OptimizeResult],
    frequency: str | None,
    func: str,
    arch_limit: int | None,
):
    plots = []
    grouped_results = itertools.groupby(results, lambda r: r.arch)
    limited_group_results = (
        itertools.islice(r, arch_limit) for _, r in grouped_results
    )
    limited_results = list(itertools.chain.from_iterable(limited_group_results))

    top_bound = np.max(
        [getattr(r.ntwk[frequency], f"max_{func}").s_re for r in limited_results]
    )
    best_top_bound = np.min(
        [getattr(r.ntwk[frequency], f"max_{func}").s_re for r in limited_results]
    )
    top_bound = np.min([top_bound, best_top_bound * 3])

    for arch, arch_results in itertools.groupby(limited_results, lambda r: r.arch):
        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
        for combination in arch_results:
            mopt.plotting.plot_with_tolerance(
                combination.ntwk[frequency], func=func, ax=ax
            )
            ax.set_ylim(bottom=1.0, top=top_bound)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.00),
            ncol=1,
            fancybox=True,
        )
        plots.append({"svg": plot_to_svg(fig), "name": str(arch)})
    return plots


def save_schematic(schema: schemdraw.Drawing) -> str:
    svg_str: str
    svg_str = schema.get_imagedata("svg").decode("utf-8")
    svg_str = re.sub('"sans"', '"sans-serif"', svg_str)
    svg_str = re.sub(r'(<svg [^>]*?) height="[^"]+" width="[^"]+"', r"\1", svg_str)

    return svg_str


def plot_pi_schematic() -> schemdraw.Drawing:
    d: schemdraw.Drawing
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2)

        elm.cables.Coax(length=2.5).right()
        elm.Dot()
        d.push()
        elm.RBox().down()
        elm.Ground(lead=False)
        d.pop()
        elm.RBox()
        elm.Dot()
        d.push()
        elm.RBox().down()
        elm.Ground(lead=False)
        d.pop()
        elm.Line(unit=0.5)
        elm.Antenna()

    return d
