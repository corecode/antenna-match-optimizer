import io
import itertools
import re
import tempfile
import threading
from http import HTTPStatus
from pathlib import Path, PurePath

import antenna_match_optimizer as mopt
import antenna_match_optimizer.plotting as mplt
import matplotlib.pyplot as plt
import numpy as np
import schemdraw
import schemdraw.elements as elm
import skrf as rf
import stopit
from flask import (
    Blueprint,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from matplotlib.figure import Figure
from werkzeug.utils import secure_filename


class OptimizeError(Exception):
    pass


example_cache = None

bp = Blueprint("match", __name__, url_prefix="/")


@bp.before_request
def set_timeout():
    timeout = float(current_app.config.get("TIMEOUT", 10))
    handler_thread = threading.current_thread().ident
    print(f"before request {timeout} {handler_thread}")

    def send_signal():
        print("timeout, sending signal")
        stopit.async_raise(handler_thread, stopit.TimeoutException)

    g.timeout_timer = threading.Timer(timeout, send_signal)
    g.timeout_timer.start()


@bp.after_request
def stop_timeout(resp):
    g.timeout_timer.cancel()
    return resp


@bp.errorhandler(stopit.TimeoutException)
def timeouterror(error):
    flash("Optimizing this file and frequency range took too long")
    return redirect(request.url, code=HTTPStatus.SEE_OTHER)


@bp.route("/")
@bp.route("/", endpoint="index")
def upload():
    pi_network = save_schematic(plot_pi_schematic())
    share_dir = current_app.config.get("SHARE_DIR", None)
    sharing_enabled = (
        share_dir is not None and (Path(current_app.root_path) / share_dir).is_dir()
    )
    return render_template(
        "upload.html",
        pi_network=pi_network,
        sharing_enabled=sharing_enabled,
    )


@bp.route("/example")
def example():
    global example_cache
    if example_cache is None:
        example_cache = optimize_internal(
            str(Path(__file__).with_name("example.s1p")),
            "Example Antenna",
            "2.4-2.5GHz",
            max_points=51,
        )
    return render_template("optimize.html", **example_cache)


@bp.route("/optimize", methods=["GET", "POST"])
def optimize():
    if request.method != "POST":
        return redirect(url_for(".upload"), code=HTTPStatus.SEE_OTHER)

    try:
        touchstone = request.files["touchstone"]
    except Exception:
        flash("No Touchstone file uploaded")
        return redirect(url_for(".upload"), code=HTTPStatus.SEE_OTHER)

    touchstone_name = PurePath(touchstone.filename or "Noname")
    touchstone_data = touchstone.read().decode("utf-8")
    touchstone_io = io.StringIO(touchstone_data)
    touchstone_io.name = touchstone_name.name

    frequency = request.form.get("frequency")
    if frequency is None or frequency == "":
        flash("You need to specify a frequency range")
        return redirect(url_for(".upload"), code=HTTPStatus.SEE_OTHER)

    try:
        template_args = optimize_internal(
            touchstone_io=touchstone_io,
            name=touchstone_name.stem,
            frequency=frequency,
            title=request.form.get("title", None),
            max_points=current_app.config.get("MAX_FREQ_POINTS", 51),
        )
    except OptimizeError as e:
        current_app.logger.error(e.__cause__)
        flash(str(e))
        return redirect(url_for(".upload"), code=HTTPStatus.SEE_OTHER)

    response = render_template("optimize.html", **template_args)

    if request.form.get("share", False):
        return store_and_redirect(touchstone_name.stem, frequency, response)
    else:
        return response


def store_and_redirect(name: str, frequency: str, response: str):
    try:
        share_dir = Path(current_app.root_path) / current_app.config["SHARE_DIR"]
        base_name = f"{name}-{frequency}"
        clean_prefix = secure_filename(base_name)

        for serial in range(1, 9999999):
            final_name = f"{clean_prefix}-{serial}"
            try:
                with open(share_dir / final_name, mode="xt") as f:
                    f.write(response)
                return redirect(
                    url_for(".share", name=final_name),
                    code=HTTPStatus.SEE_OTHER,
                )
            except FileExistsError:
                pass
        raise RuntimeError(f"could not generate unique filename for {base_name}")
    except Exception as e:
        current_app.logger.error(e)
        return response  # silent fallback


@bp.route("/share/<name>")
def share(name: str):
    return send_from_directory(
        directory=current_app.config["SHARE_DIR"],
        path=name,
        mimetype="text/html",
    )


def optimize_internal(
    touchstone_io: str | io.StringIO,
    name: str,
    frequency: str,
    title: str | None,
    max_points: int,
):
    optimize_messages: list[str] = []

    try:
        base = rf.Network(
            file=touchstone_io,
            name=name,
            f_unit="GHz",
        )
    except Exception as e:
        raise OptimizeError("Could not parse Touchstone file") from e

    if base.number_of_ports != 1:
        raise OptimizeError(
            f"Touchstone file is not a 1-port file: found {base.number_of_ports} ports"
        )

    try:
        args = mopt.OptimizerArgs(ntwk=base, frequency=frequency, max_points=max_points)
    except Exception as e:
        raise OptimizeError("Frequency range is invalid") from e

    if len(args.bandlimited_ntwk.frequency) != len(args.ntwk.frequency[args.frequency]):
        optimize_messages.append(
            f"Frequency points in optimization reduced from \
            {len(args.ntwk.frequency[args.frequency])} to \
            {len(args.bandlimited_ntwk.frequency)}"
        )

    ideal = mopt.optimize(args)
    results = mopt.evaluate_components(args, *ideal)
    best_narrow = results[0]
    best_wide = mopt.expand_result(args, results[0])

    base_smith_fig = mplt.plot_smith(base, highlight=args.bandlimited_ntwk)
    base_smith = plot_to_svg(base_smith_fig)

    base_vswr_fig = mplt.plot_vswr(args.bandlimited_ntwk)
    worst_vswr = base_vswr_fig.gca().get_ylim()[1]
    base_vswr = plot_to_svg(base_vswr_fig)

    best_smith_fig = mplt.plot_smith(best_wide.ntwk, highlight=best_narrow.ntwk)
    best_smith_fig.gca().get_legend().remove()
    best_smith = plot_to_svg(best_smith_fig)

    best_vswr_fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    mplt.plot_with_tolerance(best_narrow.ntwk, ax=ax)
    ax.set_ylim(bottom=1.0, top=worst_vswr)
    best_vswr = plot_to_svg(best_vswr_fig)

    best_schema = save_schematic(
        mplt.plot_schematic(best_narrow, antenna_name=base.name)
    )

    results_vswr = plot_architectures(
        sorted(results, key=lambda r: r.arch.value),
        func="s_vswr",
        arch_limit=3,
    )

    plt.close("all")

    return {
        "optimize_messages": optimize_messages,
        "base_name": base.name,
        "frequency": frequency,
        "title": title or base.name,
        "base_smith": base_smith,
        "base_vswr": base_vswr,
        "best_name": best_narrow.ntwk[0].name,
        "best_smith": best_smith,
        "best_vswr": best_vswr,
        "best_schema": best_schema,
        "results_vswr": results_vswr,
    }


def plot_to_svg(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    str = buf.getvalue().decode("utf-8")
    str = re.sub(r'(<svg [^>]*?) width="[^"]+" height="[^"]+"', r"\1", str)
    str = re.sub(
        r'(font: [^ ;]+) (?:[^;"]*)', r"\1 var(--pico-font-family-sans-serif)", str
    )
    str = re.sub(r"(stroke|fill): #ffffff", r"\1: var(--pico-background-color)", str)
    str = re.sub(r"(stroke|fill): #555555", r"\1: var(--pico-muted-color)", str)
    str = re.sub(r"(stroke|fill): #eeeeee", r"\1: var(--pico-muted-border-color)", str)
    str = re.sub(r"(stroke|fill): #000000", r"\1: var(--pico-color)", str)
    return str


def plot_architectures(
    results: list[mopt.OptimizeResultToleranced],
    func: str,
    arch_limit: int | None,
):
    plots = []
    grouped_results = itertools.groupby(results, lambda r: r.arch)
    limited_group_results = (
        itertools.islice(r, arch_limit) for _, r in grouped_results
    )
    limited_results = list(itertools.chain.from_iterable(limited_group_results))

    top_bound = max(
        np.max(getattr(r.ntwk, f"max_{func}").s_re) for r in limited_results
    )
    best_top_bound = min(
        np.min(getattr(r.ntwk, f"max_{func}").s_re) for r in limited_results
    )
    top_bound = np.min([top_bound, best_top_bound * 3])

    for arch, arch_results in itertools.groupby(limited_results, lambda r: r.arch):
        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
        for combination in arch_results:
            mopt.plotting.plot_with_tolerance(combination.ntwk, func=func, ax=ax)
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
    svg_str = re.sub(":black;", ":currentColor;", svg_str)
    svg_str = re.sub('="black"', '="currentColor"', svg_str)
    svg_str = re.sub(
        'font-family="sans-serif"',
        "font-family=var(--pico-font-family-sans-serif)",
        svg_str,
    )

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
