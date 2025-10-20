import numpy as np


def test_mass_per_unit_length(pinned_beam):
    """Test the mass per unit length of the beam."""
    assert beam.mass_per_unit_length == 54.0


def test_area_moment(beam):
    """Test the area moment of inertia of the beam."""
    assert np.isclose(beam.area_moment, 1.666e-5, atol=1e-8)


def test_gen_mass(pinned_beam, clamped_beam):
    """Test the generalized mass of the beam."""
    print(beam.gen_mass)
    assert np.isclose(pinned_beam.gen_mass, 27.0, atol=1e-8)
    assert np.isclose(clamped_beam.gen_mass, ######, atol=1e-8)


def test_freq(beam, modal_indx=2):
    """Test the frequency calculation of the beam."""
    freq = beam.freq(modal_indx)
    assert np.isclose(freq, 5802.78195197, atol=1e-8)


def test_freqs(beam):
    """Test the frequencies calculation of the beam."""
    assert np.all(
        np.isclose(
            beam.freqs,
            np.array([1450.69548799, 5802.78195197, 13056.2593919]),
            atol=1e-4,
        )
    )


def test_freqs_sq(beam):
    """Test the squared frequencies calculation of the beam."""
    assert np.all(
        np.isclose(
            beam.freqs_sq,
            np.array([2104517.39887, 33672278.3821, 170465909.309]),
            atol=1e-3,
        )
    )


def test_freqs_sq_mass(beam):
    """Test the squared frequencies calculation of the beam."""
    assert np.all(
        np.isclose(
            beam.freqs_sq * beam.gen_mass,
            np.array([56821969.76949, 909151516.3167, 4602579551.342999]),
            atol=1e-3,
        )
    )


def test_freq_prefactor(beam):
    """Test the frequency prefactor calculation of the beam."""
    assert np.isclose(beam.freq_prefactor, 2051.59323399, atol=1e-8)


def test_shape(beam, x=0.5, modal_indx=2):
    """Test the shape function of the beam."""
    shape_value = beam.shape(modal_indx, x)
    assert np.isclose(shape_value, 0.0, atol=1e-8)


def test_constraint_shapes(beam, constraints):
    """Test the constraint shapes calculation of the beam."""
    constraint_shapes = beam.constraint_shapes(constraints)
    expected_shapes = np.array(
        [[1.0, 0.951056516295], [0.0, 0.587785252292], [-1.0, -0.587785252292]]
    )
    assert np.all(np.isclose(constraint_shapes, expected_shapes, atol=1e-8))
