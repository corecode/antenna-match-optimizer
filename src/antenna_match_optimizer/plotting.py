import math

import matplotlib.pyplot as plt
import matplotlib.style
import schemdraw
import schemdraw.elements as elm
import skrf as rf

import antenna_match_optimizer as mopt


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


def plot_schematic(
    config: mopt.OptimizeResult, antenna_name: str = ""
) -> schemdraw.Drawing:
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

    d: schemdraw.Drawing
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

    return d
