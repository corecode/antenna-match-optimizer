import itertools

import antenna_match_optimizer.optimizer as mopt
import numpy as np
import pytest
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


def test_optimize_args_rejects_non_1port():
    twoport = rf.DefinedGammaZ0().line(1)
    with pytest.raises(ValueError):
        mopt.OptimizerArgs(ntwk=twoport, frequency="2.4-2.4835GHz")


def test_optimize_args_with_npoints():
    t_points = 21
    freq = rf.Frequency(2, 3, 1001, unit="GHz")
    ntwk = rf.DefinedGammaZ0(freq).match()

    args = mopt.OptimizerArgs(ntwk=ntwk, frequency="2.4-2.5GHz", max_points=t_points)

    assert args.bandlimited_ntwk.frequency.npoints == t_points


def test_optimize_returns_all_archs():
    detuned_ant = make_detuned_antenna()

    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    optimized = mopt.optimize(args)

    assert optimized[0].arch == mopt.Arch.Lshunt
    assert optimized[0].x[0] == approx(4.915, rel=1e-3)

    assert optimized[1].arch == mopt.Arch.Cshunt
    assert optimized[1].x[1] == approx(0.2, rel=1e-3)

    assert optimized[2].arch == mopt.Arch.Lseries
    assert optimized[2].x[0] == approx(1.218, rel=1e-3)

    assert optimized[3].arch == mopt.Arch.Cseries
    assert optimized[3].x[1] == approx(30.0, rel=1e-3)

    assert optimized[4].arch == mopt.Arch.LshuntCseries
    assert optimized[4].x[0] == approx(4.442, rel=1e-3)
    assert optimized[4].x[1] == approx(12.24, rel=1e-3)

    assert optimized[5].arch == mopt.Arch.CshuntLseries
    assert optimized[5].x[0] == approx(2.774, rel=1e-3)
    assert optimized[5].x[1] == approx(0.9583, rel=1e-3)

    assert optimized[6].arch == mopt.Arch.LseriesCshunt
    assert optimized[6].x[0] == approx(1.326, rel=1e-3)
    assert optimized[6].x[1] == approx(0.2, rel=1e-3)

    assert optimized[7].arch == mopt.Arch.CseriesLshunt
    assert optimized[7].x[0] == approx(4.613, rel=1e-3)
    assert optimized[7].x[1] == approx(30, rel=1e-3)


def test_optimize_creates_correct_name():
    detuned_ant = make_detuned_antenna()

    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    optimized = mopt.optimize(args)

    assert "4.44nH" in optimized[4].ntwk.name
    assert "12.2pF" in optimized[4].ntwk.name


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

    assert len(result) == 21


def test_evaluate_components_is_sorted():
    detuned_ant = make_detuned_antenna()
    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    minima = mopt.optimize(args)
    configs = mopt.evaluate_components(args, *minima)

    for a, b in itertools.pairwise(configs):
        asum = np.sum(a.ntwk.max_s_mag.s_mag**2)
        bsum = np.sum(b.ntwk.max_s_mag.s_mag**2)
        assert (a.arch, a.x) != (b.arch, b.x)
        assert asum < bsum

    assert configs[0].arch == mopt.Arch.LshuntCseries
    assert configs[0].x == (4.7, 15.0)


def test_expand_result_single():
    detuned_ant = make_detuned_antenna()
    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    minima = mopt.optimize(args)

    result = mopt.expand_result(args, minima[0])

    assert result.ntwk.frequency == detuned_ant.frequency  # type: ignore
    assert result.ntwk.name == minima[0].ntwk.name


def test_expand_result_set():
    detuned_ant = make_detuned_antenna()
    args = mopt.OptimizerArgs(ntwk=detuned_ant, frequency="2.4-2.4835GHz")
    minima = mopt.optimize(args)
    configs = mopt.evaluate_components(args, minima[2])

    result = mopt.expand_result(args, configs[0])

    assert result.ntwk[0].frequency == detuned_ant.frequency  # type: ignore
    assert result.ntwk.name == configs[0].ntwk.name
    assert result.ntwk[0].name == configs[0].ntwk[0].name
    assert result.ntwk[0] != result.ntwk[1]
