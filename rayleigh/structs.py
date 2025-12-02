import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
from functools import cached_property
from constants import (
    valid_beam_boundary_conditions,
    valid_plate_boundary_conditions,
)
from clampedSolver import betaL_roots
import matplotlib.pyplot as plt

class Beam:
    """Class that stores all structural information about a sample beam."""

    def __init__(
        self,
        length: float,
        e_modulus: float,
        density: float,
        thickness: float,
        width: float,
        boundary_condition: str,
        modal_indx: int,
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
            Note that the modal_indx is one based indexing!!!

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

        # Validate boundary condition
        if self.boundary_condition not in valid_beam_boundary_conditions:
            raise ValueError(
                f"""Invalid boundary condition '{self.boundary_condition}'
                for Beam."""
                f"Valid options are: {valid_beam_boundary_conditions}"
            )

    # Here we calculate derived properties, cached properties allow us to
    # avoid recalculating them multiple times. We calculate once
    # then store the result.

    @cached_property
    def indx(self) -> int:
        return self.modal_indx

    @cached_property
    def mass_per_unit_length(self) -> float:
        """🐍Calculate the mass per unit length of the beam. Units of kg / m."""
        return self.density * self.thickness * self.width

    @cached_property
    def area_moment(self) -> float:
        """🐍Calculate the area moment of inertia for the beam. Units of m^4."""
        return self.width * self.thickness**3 / 12

    def gen_mass(self, modal_indx) -> float:
        """🐍Calculate the generalized mass of the beam based on its
        boundary condition. Units of kg."""
        if self.boundary_condition == "PP":
            return self.length * self.mass_per_unit_length / 2
        elif self.boundary_condition == "CC":
            I, _ = quad(lambda x: self.psi_beam(modal_indx, x)**2, 0.0, self.length)
            return self.mass_per_unit_length * I
        
    @cached_property
    def gen_mass_vector_beam(self) -> float:
        return np.array([self.gen_mass(n) for n in range(1, self.modal_indx + 1)])

    @cached_property
    def freqs(self) -> np.ndarray:
        """🐍Calculate the first indx modal frequencies of the beam.
        Units of rad/sec."""
        return np.array([self.freq(n) for n in range(1, self.modal_indx + 1)])

    @cached_property
    def freqs_sq(self) -> np.ndarray:
        """🐍Calculate the square of the frequencies. Units of rad^2/sec^2."""
        return self.freqs**2

    @cached_property
    def freqs_sq_mass(self) -> np.ndarray:
        """🐍Calculate the square of the frequencies multiplied by the
        generalized mass. Units of kg * rad^2/sec^2."""
        return self.freqs_sq * self.gen_mass_vector_beam    

    @cached_property
    def freq_prefactor(self) -> float:
        """🐍Calculate the frequency prefactor for the beam.
        Units of s^-1 * m^(-1/2)."""
        return (
            np.sqrt(self.e_modulus * self.area_moment / self.mass_per_unit_length)
            * np.pi**2
            / self.length**2
        )
    
    def psi_beam(self, modal_indx, x):
        xi = x / self.length
        i = modal_indx - 1
        k = betaL_roots[i]

        sigma = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
        z = k * xi
        return (np.cosh(z) - np.cos(z) - sigma * (np.sinh(z) - np.sin(z)))
                
    def shape(self, modal_indx, x) -> float:
        """🐍Calculate the shape function for the beam at a given position.
        Units of m.

        Args:
            modal_indx (int): modal index of the beam
            x (float): position along the beam in meters

        Returns:
            float : The value of the shape function at the given position.
        """

        if self.boundary_condition == "PP":
            return np.sin(modal_indx * np.pi * x / self.length)
        elif self.boundary_condition == "CC":
            grid = np.linspace(0, self.length, 4001)
            A = np.max(np.abs(self.psi_beam(modal_indx, grid)))
            return (self.psi_beam(modal_indx, x) / A)

    def freq(self, modal_indx: float) -> float:
        """🐍Calculate the frequency of the beam for a given modal index.
        Units of rad/sec."""
        if self.boundary_condition == "PP":
            return (
                np.sqrt(
                    self.e_modulus
                    * self.area_moment
                    / self.mass_per_unit_length
                )
                * (modal_indx * np.pi / self.length) ** 2
            )
        
        elif self.boundary_condition == "CC":
            i = modal_indx - 1
            k = betaL_roots[i]
            return (
                np.sqrt(
                    self.e_modulus * self.area_moment / self.mass_per_unit_length
                )
                * (k / self.length) ** 2
            )

    def constraint_shapes(self, constraints: np.ndarray) -> np.ndarray:
        """🐍Calculate the constraint shapes for the beam.

        Args:
            constraints (n x 2): array of constraints where each row is a
            constraint and column 0 is x index and column 1 is y index.

        Returns:
            shapes (n x m): array of shape functions evaluated at the
            constraints. Each row corresponds to a mode and each column
            corresponds to a constraint.
        """

        shapes = np.zeros((self.modal_indx, len(constraints)))

        for n in range(1, self.modal_indx + 1):
            for indx, constraint in enumerate(constraints):
                shapes[n - 1][indx] = self.shape(n, constraint[0])
        return shapes

    def make_constraint_block(self, constraints) -> np.ndarray:
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
        x_length,
        y_length,
        e_modulus,
        density,
        thickness,
        poisson_ratio,
        boundary_condition,
        x_modal_indx,
        y_modal_indx,
    ):
        """Initializer of the class to set up the plate properties.

        Args:
            x_length (float): length of the plate in the x direction
            y_length (float): length of the plate in the y direction
            e_modulus (float): elastic modulus of the plate in N/m
            density (float): density of the plate in kg/m^3
            thickness (float): thickness of the plate in m
            possion_ratio (float): Poisson's ratio of the plate
            boundary_condition (string): boundary condition
            x_modal_indx (float): number of modes in the x direction
            y_modal_indx (float): number of modes in the y direction

        Returns:
            Implicitly returns the plate object with all its properties set.

        Raises:
            Exception: If an unknown boundary condition is provided.
        """
        self.x_length = x_length
        self.y_length = y_length
        self.e_modulus = e_modulus
        self.density = density
        self.thickness = thickness
        self.poisson_ratio = poisson_ratio
        self.boundary_condition = boundary_condition
        self.x_modal_indx = x_modal_indx
        self.y_modal_indx = y_modal_indx

        # Validate boundary condition
        if self.boundary_condition not in valid_plate_boundary_conditions:
            raise ValueError(
                f"Invalid boundary condition '{self.boundary_condition}' for Plate. "
                f"Valid options are: {valid_plate_boundary_conditions}"
            )

    @cached_property
    def x_indx(self) -> int:
        return self.x_modal_indx
    
    @cached_property
    def y_indx(self) -> int:
        return self.y_modal_indx
    
    @cached_property
    def mass_per_unit_area(self):
        """Calculate the mass per unit area of the plate."""
        return self.density * self.thickness

    @cached_property
    def flexural_rigidity(self):
        """Calculate the flexural rigidity of the plate."""
        return (
            self.e_modulus
            * self.thickness**3
            / (12 * (1 - self.poisson_ratio**2))
        )

    @cached_property
    def freq_prefactor(self):
        """Calculate the frequency prefactor for the plate."""
        return self.flexural_rigidity * np.pi**4 / (self.mass_per_unit_area)

    def gen_mass(self, rx, ry):
        """Calculate the generalized mass of the plate based on its
        boundary condition."""
        if self.boundary_condition == "PPPP":
            return self.mass_per_unit_area * self.x_length * self.y_length / 4
        elif self.boundary_condition == "CCCC":
            def integrand(y, x):
                return self.shape(rx, ry, x, y)**2
            I, _ = dblquad(
                integrand,
                0.0, self.y_length,
                lambda y: 0.0,
                lambda y: self.x_length)
            return self.mass_per_unit_area * I

    @cached_property
    def gen_mass_vector_plate(self) -> float:
        masses = np.zeros((self.x_modal_indx, self.y_modal_indx))
        for rx in range(1, self.x_modal_indx + 1):
            for ry in range(1, self.y_modal_indx + 1):
                masses[rx - 1, ry - 1] = self.gen_mass(rx, ry)
        return masses

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
        return self.freqs_sq * self.gen_mass_vector_plate

    def shape(self, rx, ry, x, y):
        """Calculate the shape function for the plate at a given position.

        Args:
            rx (int): x modal index of the plate
            ry (int): y modal index of the plate
            position (1 x 2 array of floats): x, y position in meters

        Returns:
            float : The value of the shape function at the given position.
        """
        if self.boundary_condition == "PPPP":
            return np.sin(rx * np.pi * x / self.x_length) * np.sin(
                ry * np.pi * y / self.y_length
            )
        elif self.boundary_condition == "CCCC":
            grid_x = np.linspace(0, self.x_length, 4001)
            A = np.max(np.abs(self.psi_plate(rx, grid_x)))
            grid_y = np.linspace(0, self.y_length, 4001)
            B = np.max(np.abs(self.psi_plate(ry, grid_y)))
            
            xi_x = x / self.x_length
            xi_y = y / self.y_length
            sx = self.psi_plate(rx, xi_x) / A
            sy = self.psi_plate(ry, xi_y) / B
            return sx * sy

    def freq(self, rx, ry):
        """Calculate the frequency of the plate for given modal indices. Units of rad/sec."""
        if self.boundary_condition == "PPPP":
            return np.sqrt(self.flexural_rigidity / self.gen_mass(rx, ry)) * (
                (rx * np.pi / self.x_length) ** 2
                + (ry * np.pi / self.y_length) ** 2
            )
        elif self.boundary_condition == "CCCC":
            D = (self.e_modulus * self.thickness**3) / (12 * (1 - self.poisson_ratio**2))
            G = 1.506
            H = 1.248
            J = 1.248
            aspect_ratio = self.x_length / self.y_length

            C = (np.pi**4 * D) / (self.x_length**4 * self.mass_per_unit_area)

            return (
                C * (
                    G**4
                    + G**4 * aspect_ratio**4
                    + 2 * aspect_ratio**2 * (
                        self.poisson_ratio * H * J
                        + (1 - self.poisson_ratio) * H * J
                    )
                )
            )**0.5

    def constraint_shapes(self, constraints):
        shapes = np.zeros(
            (self.x_modal_indx, self.y_modal_indx, len(constraints))
        )

        for rx in range(1, self.x_modal_indx + 1):
            for ry in range(1, self.y_modal_indx + 1):
                for indx, constraint in enumerate(constraints):
                    shapes[rx - 1, ry - 1, indx] = self.shape(rx, ry, constraint[0], constraint[1])
        return shapes

    def psi_plate(self, modal_indx, x):
        i = modal_indx - 1
        k = betaL_roots[i]

        sigma = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
        z = k * x
        return np.cosh(z) - np.cos(z) - sigma * (np.sinh(z) - np.sin(z))