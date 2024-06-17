import antenna_match_optimizer.plotting as mplt
import skrf as rf


def make_antenna() -> rf.Network:
    return rf.Network("tests/2450AT18A100.s1p")


def test_plot_smith_highlight():
    ant = make_antenna()
    mplt.plot_smith(ant, highlight=ant["2.4-2.5GHz"])


def test_plot_smith_networkset():
    ant = make_antenna()
    nset = rf.NetworkSet([ant, ant])
    mplt.plot_smith(nset, highlight=nset["2.4-2.5GHz"])


def test_plot_vswr():
    ant = make_antenna()
    mplt.plot_vswr(ant)


def test_plot_vswr_networkset():
    ant = make_antenna()
    nset = rf.NetworkSet([ant, ant])
    mplt.plot_vswr(nset)


def test_plot_with_tolerance():
    ant = make_antenna()
    nset = rf.NetworkSet([ant, ant])
    mplt.plot_with_tolerance(nset)
