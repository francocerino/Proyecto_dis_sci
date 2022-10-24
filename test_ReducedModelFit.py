from scipy.integrate import odeint
import numpy as np
from skreducedmodel import ReducedModel
# from scipy.special import jv as BesselJ


def pend(y, t, b, λ):
    θ, ω = y
    dydt = [ω, -b*ω - λ*np.sin(θ)]

    return dydt


def test_ReducedModelFit():

    b = 0.2
    y0 = [np.pi/2, 0.]

    param = np.linspace(1, 5, 101)
    times = np.linspace(0, 50, 1001)

    training = []
    for λ in param:
        sol = odeint(pend, y0, times, (b, λ))
        training.append(sol[:, 0])

    training_set = np.array(training)
    parameters = param
    physical_points = times
    nmax = 10

    model = ReducedModel(
                     seed_global_rb=0,
                     greedy_tol=1e-16,
                     lmax=1,
                     nmax=nmax,
                     normalize=True
                     )

    rb = model.fit(
               training_set=training_set,
               parameters=parameters,
               physical_points=physical_points)

    print(rb.errors[nmax-1], rb.errors[0])

    assert rb.errors[0] > rb.errors[nmax-1]
    assert rb.errors[5] > rb.errors[nmax-1]
    assert len(rb.indices) == nmax
    assert len(rb.indices) == nmax
    # todos los numeros salieron del ejemplo del Pendulo
    assert rb.indices[9] == 92


def test_rmfit_parameters():

    b = 0.2
    y0 = [np.pi/2, 0.]

    param = np.linspace(1, 5, 101)
    times = np.linspace(0, 50, 1001)

    training = []
    for λ in param:
        sol = odeint(pend, y0, times, (b, λ))
        training.append(sol[:, 0])

    training_set = np.array(training)
    parameters = param
    physical_points = times
    # nmax = 10

    model1 = ReducedModel(
                    seed_global_rb=0,
                    greedy_tol=1e-1,
                    lmax=1,
                    )

    model2 = ReducedModel(
                    seed_global_rb=0,
                    greedy_tol=1e-16,
                    lmax=1,
                    )

    rb1 = model1.fit(
            training_set=training_set,
            parameters=parameters,
            physical_points=physical_points,
            )

    rb2 = model2.fit(
            training_set=training_set,
            parameters=parameters,
            physical_points=physical_points,
            )

    assert len(rb1.indices) < len(rb2.indices)


def test_rom_rb_interface(rom_parameters):
    """Test API consistency."""
    training_set = rom_parameters["training_set"]
    physical_points = rom_parameters["physical_points"]
    parameter_points = rom_parameters["parameter_points"]

    model = ReducedModel(greedy_tol=1e-14)

    bessel = model.fit(training_set=training_set,
                       physical_points=physical_points,
                       parameters=parameter_points
                       )

    # bessel = ReducedOrderModel(
    #    training_set, physical_points, parameter_points, greedy_tol=1e-14
    # )
    basis = bessel.basis.data
    errors = bessel.errors
    projection_matrix = bessel.projection_matrix
    greedy_indices = bessel.indices
    # eim = bessel.eim_

    assert len(basis) == 10
    assert len(errors) == 10
    assert len(projection_matrix) == 101
    assert len(greedy_indices) == 10
    # assert eim == bessel.basis_.eim
