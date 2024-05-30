import skrf as rf
from pytest import approx

import src.antenna_match_optimizer as mopt


def make_detuned_antenna():
    ant = rf.Network("tests/2450AT18A100.s1p")
    line = rf.DefinedGammaZ0(frequency=ant.frequency)
    # random and unscientific perturbation
    ant_detune = (
        line.shunt_capacitor(0.5e-12)
        ** line.inductor(0.1e-9)
        ** line.shunt_capacitor(0.5e-12)
        ** ant
    )
    return ant_detune


def test_optimize_returns_all_archs():
    detuned_ant = make_detuned_antenna()

    optimized = mopt.optimize(ntwk=detuned_ant, frequency="2.4-2.4835GHz")

    assert optimized[0].arch == mopt.Arch.LshuntCseries
    assert optimized[0].x[0] == approx(4.442, rel=1e-3)
    assert optimized[0].x[1] == approx(12.24, rel=1e-3)

    assert optimized[1].arch == mopt.Arch.CshuntLseries
    assert optimized[1].x[0] == approx(2.773, rel=1e-3)
    assert optimized[1].x[1] == approx(0.9583, rel=1e-3)

    assert optimized[2].arch == mopt.Arch.LseriesCshunt
    assert optimized[2].x[0] == approx(1.219, rel=1e-3)
    assert optimized[2].x[1] == approx(1e-3, rel=1e-3)

    assert optimized[3].arch == mopt.Arch.CseriesLshunt
    assert optimized[3].x[0] == approx(4.761, rel=1e-3)
    assert optimized[3].x[1] == approx(60, rel=1e-3)
