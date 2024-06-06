import itertools

import antenna_match_optimizer as mopt
import numpy as np
import skrf as rf
from pytest import approx


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
    ant_detune.name = "ANT"
    return ant_detune


def test_optimize_returns_all_archs():
    detuned_ant = make_detuned_antenna()

    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    optimized = mopt.optimize(args)

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


def test_optimize_creates_correct_name():
    detuned_ant = make_detuned_antenna()

    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    optimized = mopt.optimize(args)

    assert "4.44nH" in optimized[0].ntwk.name
    assert "12.2pF" in optimized[0].ntwk.name


def test_closest_values_exact():
    result = mopt.closest_values(1.001, np.array(((0.9, 0.1), (1.0, 0.1), (1.1, 0.1))))

    np.testing.assert_array_equal(result, [(1.0, 0.1)])


def test_closest_values():
    result = mopt.closest_values(0.95, np.array(((0.9, 0.1), (1.0, 0.1), (1.1, 0.1))))

    np.testing.assert_array_equal(result, [(0.9, 0.1), (1.0, 0.1)])


def test_closest_values_one_sided():
    result = mopt.closest_values(
        1.0, np.array(((0.4, 0.1), (0.9, 0.1), (0.8, 0.1), (1.5, 0.1), (1.6, 0.2)))
    )

    np.testing.assert_array_equal(result, [(0.9, 0.1), (0.8, 0.1), (1.5, 0.1)])


def test_closest_values_below_bound():
    result = mopt.closest_values(
        0.001,
        np.array(((0.4, 0.1), (0.9, 0.1), (0.8, 0.1), (1.5, 0.1), (1.6, 0.2))),
    )

    np.testing.assert_array_equal(result, [(0.001, 0.0)])


def test_closest_values_above_bound():
    result = mopt.closest_values(
        3.2, np.array(((0.4, 0.1), (0.9, 0.1), (0.8, 0.1), (1.5, 0.1), (1.6, 0.2)))
    )

    np.testing.assert_array_equal(result, [(3.2, 0.0)])


def test_expand_tolerance():
    result = mopt.expand_tolerance((2.7, 0.2))

    np.testing.assert_allclose(result, [2.7, 2.5, 2.9])


def test_component_combinations_creates_component_product():
    result = mopt.component_combinations(
        arch=mopt.Arch.LseriesCshunt,
        x=(1.2, 1.1),
        inductors=np.array([[1.0, 0.0], [1.3, 0.0]]),
        capacitors=np.array([[1.0, 0.0], [1.2, 0.0]]),
    )

    assert list(result) == approx(
        [
            ((mopt.Arch.LseriesCshunt, (1.3, 1.2)), (1.3, 1.2)),
            ((mopt.Arch.LseriesCshunt, (1.3, 1.0)), (1.3, 1.0)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.2)), (1.0, 1.2)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.0, 1.0)),
        ]
    )


def test_component_combinations_creates_tolerance_product():
    result = mopt.component_combinations(
        arch=mopt.Arch.LseriesCshunt,
        x=(1.2, 1.1),
        inductors=np.array([[1.0, 0.1]]),
        capacitors=np.array([[1.0, 0.1]]),
    )

    assert list(result) == approx(
        [
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.0, 1.0)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.0, 0.9)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.0, 1.1)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (0.9, 1.0)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (0.9, 0.9)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (0.9, 1.1)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.1, 1.0)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.1, 0.9)),
            ((mopt.Arch.LseriesCshunt, (1.0, 1.0)), (1.1, 1.1)),
        ]
    )


def test_evaluate_components_unlimited():
    detuned_ant = make_detuned_antenna()
    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    optimized = mopt.optimize(args)

    result = mopt.evaluate_components(args, *optimized)

    assert len(result) == 15


def test_evaluate_components_is_sorted():
    detuned_ant = make_detuned_antenna()
    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    minima = mopt.optimize(args)
    configs = mopt.evaluate_components(args, *minima)

    for a, b in itertools.pairwise(configs):
        assert np.sum(a.ntwk.max_s_mag.s_mag**2) < np.sum(b.ntwk.max_s_mag.s_mag**2)

    assert configs[0].arch == mopt.Arch.LshuntCseries
    assert configs[0].x == (4.7, 15.0)


def test_expand_result_single():
    detuned_ant = make_detuned_antenna()
    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    minima = mopt.optimize(args)

    result = mopt.expand_result(args, minima[0])

    assert result.ntwk.frequency == detuned_ant.frequency
    assert result.ntwk.name == minima[0].ntwk.name


def test_expand_result_set():
    detuned_ant = make_detuned_antenna()
    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    minima = mopt.optimize(args)
    configs = mopt.evaluate_components(args, minima[2])

    result = mopt.expand_result(args, configs[0])

    assert result.ntwk[0].frequency == detuned_ant.frequency
    assert result.ntwk.name == configs[0].ntwk.name
    assert result.ntwk[0].name == configs[0].ntwk[0].name
    assert result.ntwk[0] != result.ntwk[1]
