import pytest


# Fixtures for structs tests
@pytest.fixture
def beam():
    from rayleigh.structs import Beam

    return Beam(
        length=1.0,
        e_modulus=70e9,
        thickness=0.1,
        width=0.2,
        density=2700,
        boundary_condition="PP",
        modal_indx=3,
    )


@pytest.fixture
def plate():
    from rayleigh.structs import Plate

    return Plate(
        x_length=1.0,
        y_length=2.0,
        e_modulus=70e9,
        density=2700,
        thickness=0.01,
        poisson_ratio=0.3,
        boundary_condition="PPPP",
        x_modal_indx=3,
        y_modal_indx=2,
    )


@pytest.fixture
def constraints():
    import numpy as np

    return np.array([[0.5, 0.4], [0.4, 0.3]])
