import numpy as np
from functools import cached_property


class Beam:
    """Class that stores all structural information about a sample beam."""

    def __init__(
        self,
        length,
        e_modulus,
        thickness,
        width,
        density,
        boundary_condition,
        modal_indx,
    ):
        """Initializer of the class to set up the beam properties.

        Args:
            length (float): length of the beam in meters
            e_modulus (float): elastic modulus of the beam in N/m
            density (float): density of the beam in kg/m^3
            thickness (float): beam thickness in m
            width (float): beam width in m
            boundary_condition (string): boundary condition
            modal_indx (float): number of modes requested for the beam

        Returns:
            Implicitly returns the beam object with all its properties set.

        Raises:
            Exception: If an unknown boundary condition is provided.
        """
        self.length = length
        self.e_modulus = e_modulus
        self.thickness = thickness
        self.width = width
        self.density = density
        self.boundary_condition = boundary_condition
        self.modal_indx = modal_indx

        # Here we calculate derived properties, cached properties allow us to
        # avoid recalculating them multiple times. We calculate once
        # then store the result.

        @cached_property
        def mass_per_unit_length(self):
            """Calculate the mass per unit length of the beam."""
            return self.density * self.thickness * self.width

        @cached_property
        def area_moment(self):
            """Calculate the area moment of inertia for the beam."""
            return self.width * self.thickness**3 / 12

        @cached_property
        def gen_mass(self):
            """Calculate the generalized mass of the beam based on its
            boundary condition."""
            if self.boundary_condition == "PP":
                return self.length * self.mass_per_unit_length / 2
            else:
                print("An error occurred, unknown BC Beam condition.")
                raise Exception(
                    "An error occurred, unknown BC Beam condition."
                )

        @cached_property
        def freqs(self):
            """Calculate the first indx modal frequencies of the beam."""
            freqs = np.zeros_like(self.modal_indx)
            for n in range(1, self.modal_indx + 1):
                freqs[n - 1] = self.freq(n)
            return freqs

        @cached_property
        def freqs_sq(self):
            """Calculate the square of the frequencies."""
            return self.freqs**2

        @cached_property
        def freqs_sq_mass(self):
            """Calculate the square of the frequencies multiplied by the
            generalized mass."""
            return self.freqs_sq() * self.gen_mass

        @cached_property
        def freq_prefactor(self):
            """Calculate the frequency prefactor for the beam."""
            return (
                np.sqrt(self.e_modulus * self.area_moment / self.gen_mass)
                * np.pi**2
                / self.length**2
            )

    def shape(self, modal_indx, x):
        """Calculate the shape function for the beam at a given position.

        Args:
            modal_indx (int): modal index of the beam
            x (float): position along the beam in meters

        Returns:
            float : The value of the shape function at the given position.
        """
        if self.BC == "PP":
            return np.sin(modal_indx * np.pi * x / self.length)

    def freq(self, modal_indx):
        """Calculate the frequency of the beam for a given modal index."""
        if self.BC == "PP":
            return (
                np.sqrt(
                    self.e_modulus
                    * self.area_moment
                    / self.mass_per_unit_length
                )
                * (modal_indx * np.pi / self.length) ** 2
            )

    def constraint_shapes(self, constraints):
        """Calculate the constraint shapes for the beam.

        Args:
            constraints (n x 2): array of constraints where each row is a
            constraint and column 0 is x index and column 1 is y index.

        Returns:
            shapes (n x m): array of shape functions evaluated at the
            constraints. Each row corresponds to a modal index and each column
            corresponds to a constraint.
        """

        shapes = np.zeros((self.indx, len(constraints)))

        for n in range(1, self.indx + 1):
            for indx, constraint in enumerate(constraints):
                shapes[n - 1][indx] = self.shape(n, constraint[0])
        return shapes

    def make_constraint_block(self, constraints):
        constraint_block = np.zeros(
            (self.indx, len(constraints), len(constraints))
        )

        for row_indx in range(len(constraints)):
            for col_indx in range(len(constraints)):

                constraint_block[:, row_indx, col_indx] = (
                    self.constraint_eval[:, row_indx]
                    * self.constraint_eval[:, col_indx]
                )

        return constraint_block


class Plate:
    """Class that stores all structural information about a sample plate."""

    def __init__(
        self,
        a_length,
        b_length,
        e_modulus,
        density,
        thickness,
        nu,
        boundary_condition,
        x_modal_indx,
        y_modal_indx,
    ):
        """Initializer of the class to set up the plate properties.

        Args:
            a_length (float): length of the plate in the x direction
            b_length (float): length of the plate in the y direction
            e_modulus (float): elastic modulus of the plate in N/m
            density (float): density of the plate in kg/m^3
            thickness (float): thickness of the plate in m
            nu (float): Poisson's ratio of the plate
            boundary_condition (string): boundary condition
            x_modal_indx (float): number of modes in the x direction
            y_modal_indx (float): number of modes in the y direction

        Returns:
            Implicitly returns the plate object with all its properties set.

        Raises:
            Exception: If an unknown boundary condition is provided.
        """
        self.a_length = a_length
        self.b_length = b_length
        self.e_modulus = e_modulus
        self.density = density
        self.thickness = thickness
        self.nu = nu
        self.boundary_condition = boundary_condition
        self.x_modal_indx = x_modal_indx
        self.y_modal_indx = y_modal_indx

        @cached_property
        def mass_per_unit_area(self):
            """Calculate the mass per unit area of the plate."""
            return self.den * self.h

        @cached_property
        def flexural_rigidity(self):
            """Calculate the flexural rigidity of the plate."""
            return self.E * self.h**3 / (12 * (1 - self.nu**2))

        @cached_property
        def freq_prefactor(self):
            """Calculate the frequency prefactor for the plate."""
            return (
                self.flexural_rigidity * np.pi**4 / (self.mass_per_unit_area)
            )

        @cached_property
        def gen_mass(self):
            """Calculate the generalized mass of the plate based on its
            boundary condition."""
            if self.BC == "PPPP":
                return (
                    self.mass_per_unit_area * self.a_length * self.b_length / 4
                )
            else:
                print("An error occurred, unknown BC Plate condition.")
                raise Exception(
                    "An error occurred, unknown BC Plate condition."
                )

        @cached_property
        def freqs(self):
            """Calculate the first modal frequencies of the plate."""
            freqs = np.zeros((self.x_modal_indx, self.y_modal_indx))
            for rx in range(1, self.x_modal_indx + 1):
                for ry in range(1, self.y_modal_indx + 1):
                    freqs[rx - 1, ry - 1] = self.freq(rx, ry)
            return freqs

        @cached_property
        def freqs_sq(self):
            """Calculate the square of the frequencies."""
            return self.freqs**2

        @cached_property
        def freqs_sq_mass(self):
            """Calculate the square of the frequencies multiplied by the
            generalized mass."""
            return self.freqs_sq * self.gen_mass

    def shape(self, rx, ry, position):
        """Calculate the shape function for the plate at a given position.

        Args:
            rx (int): x modal index of the plate
            ry (int): y modal index of the plate
            position (1 x 2 array of floats): x, y position in meters

        Returns:
            float : The value of the shape function at the given position.
        """
        if self.BC == "PPPP":
            return np.sin(rx * np.pi * position[0] / self.a_length) * np.sin(
                ry * np.pi * position[1] / self.b_length
            )

    def freq(self, rx, ry):
        return np.sqrt(self.flexural_rigidity / self.generalized_mass) * (
            (rx * np.pi / self.a_length) ** 2
            + (ry * np.pi / self.b_length) ** 2
        )

    def constraint_shapes(self, constraints):
        shapes = np.zeros(
            (self.x_modal_indx, self.y_modal_indx, len(constraints))
        )

        for rx in range(1, self.x_indx + 1):
            for ry in range(1, self.y_indx + 1):
                for indx, constraint in enumerate(constraints):

                    shapes[rx - 1, ry - 1, indx] = self.shape(
                        rx, ry, constraint
                    )
        return shapes
