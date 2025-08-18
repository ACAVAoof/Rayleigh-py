import numpy as np
import pytest


def test_mass_per_unit_length(beam):
    """Test the mass per unit length of the beam."""
    assert beam.mass_per_unit_length == 54.0


def test_area_moment(beam):
    """Test the area moment of inertia of the beam."""
    assert np.isclose(beam.area_moment, 1.666e-5, atol=1e-8)


def test_gen_mass(beam):
    """Test the generalized mass of the beam."""
    print(beam.gen_mass)
    assert np.isclose(beam.gen_mass, 27.0, atol=1e-8)

    # Test with invalid boundary condition

    beam.__dict__.pop("gen_mass", None)  # Clear cached property
    beam.boundary_condition = "CC"

    with pytest.raises(ValueError):
        beam.gen_mass
