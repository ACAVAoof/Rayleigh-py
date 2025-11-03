import numpy as np


def test_mass_per_unit_area(plate):
    """Test the mass per unit area of the plate."""
    assert plate.mass_per_unit_area == 27.0


def test_flexural_rigidity(plate):
    """Test the flexural rigidity of the plate."""
    assert np.isclose(plate.flexural_rigidity, 6410.256410256412, atol=1e-8)


def test_gen_mass(plate):
    """Test the generalized mass of the plate."""
    assert np.isclose(plate.gen_mass, 13.5, atol=1e-8)


def test_freq(plate, x_modal_indx=2, y_modal_indx=3):
    """Test the frequency calculation of the plate."""
    freq = plate.freq(x_modal_indx, y_modal_indx)
    assert np.isclose(freq, 1344.1587989475395, atol=1e-8)


def test_freqs(plate):
    """Test the frequencies calculation of the plate."""
    assert np.all(
        np.isclose(
            plate.freqs,
            np.array(
                [
                    1450.69548799,
                    2901.39097598,
                    4352.08646397,
                    5802.78195197,
                    7253.47743996,
                    8704.17292795,
                ]
            ),
            atol=1e-4,
        )
    )


def test_freqs_sq(plate):
    """Test the squared frequencies calculation of the plate."""
    assert np.all(
        np.isclose(
            plate.freqs_sq,
            np.array(
                [
                    2104517.39887,
                    8422069.59548,
                    18940656.5908,
                    33672278.3821,
                    57816445.9803,
                    90373159.3847,
                ]
            ),
            atol=1e-3,
        )
    )


def test_freqs_sq_mass(plate):
    """Test the squared frequencies calculation of the plate."""
    assert np.all(
        np.isclose(
            plate.freqs_sq * plate.gen_mass,
            np.array(
                [
                    28461974.7848,
                    113695823.149,
                    255248554.042,
                    460257955.342,
                    780024101.733,
                    1215023031.19,
                ]
            ),
            atol=1e-3,
        )
    )


def test_freq_prefactor(plate):
    """Test the frequency prefactor calculation of the plate."""
    assert np.isclose(plate.freq_prefactor, 2051.59323399, atol=1e-8)


def test_shape(plate, x=0.7, y=0.5, x_modal_indx=2, y_modal_indx=3):
    """Test the shape function of the plate."""
    shape_value = plate.shape(x_modal_indx, y_modal_indx, [x, y])
    assert np.isclose(shape_value, -0.672498511964, atol=1e-8)


def test_constraint_shapes(plate, constraints):
    """Test the constraint shapes calculation of the plate."""
    shapes = plate.constraint_shapes(constraints)
    assert np.all(
        np.isclose(
            shapes,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            atol=1e-8,
        )
    )
