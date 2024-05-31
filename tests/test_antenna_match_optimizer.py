import numpy as np
import skrf as rf
from pytest import approx

import src.antenna_match_optimizer as mopt


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
    return ant_detune


def test_optimize_returns_all_archs() -> None:
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


def test_closest_values_exact() -> None:
    result = mopt.closest_values(1.001, np.array(((0.9, 0.1), (1.0, 0.1), (1.1, 0.1))))

    np.testing.assert_array_equal(result, [(1.0, 0.1)])


def test_closest_values() -> None:
    result = mopt.closest_values(0.95, np.array(((0.9, 0.1), (1.0, 0.1), (1.1, 0.1))))

    np.testing.assert_array_equal(result, [(0.9, 0.1), (1.0, 0.1)])


def test_closest_values_one_sided() -> None:
    result = mopt.closest_values(
        1.0, np.array(((0.4, 0.1), (0.9, 0.1), (0.8, 0.1), (1.5, 0.1), (1.6, 0.2)))
    )

    np.testing.assert_array_equal(result, [(0.9, 0.1), (0.8, 0.1), (1.5, 0.1)])


def test_expand_tolerance() -> None:
    result = mopt.expand_tolerance((2.7, 0.2))

    np.testing.assert_allclose(result, [2.7, 2.5, 2.9])

def test_component_combinations_creates_component_product() -> None:
    result = mopt.component_combinations(
        arch=mopt.Arch.LseriesCshunt,
        x=[1.2, 1.1],
        inductors=np.array([[1.0, 0.0],
                            [1.3, 0.0]]),
        capacitors=np.array([[1.0, 0.0],
                             [1.2, 0.0]])
        )

    assert list(result) == approx([
        ((mopt.Arch.LseriesCshunt, (1.3, 1.2)), (1.3, 1.2)),
        ((mopt.Arch.LseriesCshunt, (1.3, 1.0)), (1.3, 1.0)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.2)), (1.0, 1.2)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.0, 1.0)),
        ])

def test_component_combinations_creates_tolerance_product() -> None:
    result = mopt.component_combinations(
        arch=mopt.Arch.LseriesCshunt,
        x=[1.2, 1.1],
        inductors=np.array([[1.0, 0.1]]),
        capacitors=np.array([[1.0, 0.1]])
        )

    assert list(result) == approx([
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.0, 1.0)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.0, 0.9)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.0, 1.1)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (0.9, 1.0)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (0.9, 0.9)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (0.9, 1.1)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.1, 1.0)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.1, 0.9)),
        ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.1, 1.1)),
        ])
