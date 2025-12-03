from solver import sym_transform
from solver import asym_solve
from calculateMatrix import construct_matrix_maybefast_tiny
from solver import rayleigh_solve
from lambda_finder import calc_lambda
from lambda_finder import calc_qbar_plate
from lambda_finder import calc_modeshape_matrix
from lambda_finder import heatmap_plot

from solver import bisection_method

from structs import Beam
from structs import Plate
import time
from scipy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
import cProfile

beam1 = Beam(np.sqrt(2), 70e9, 2700, 0.001, 0.01, "CC", 5)
plate1 = Plate(np.sqrt(2), np.sqrt(2) / 2, 70e9, 2700, 0.01, 0.33, "CCCC", 5, 5)

print("SYSTEM NUMBER", beam1.e_modulus * beam1.area_moment / beam1.mass_per_unit_length * plate1.mass_per_unit_area / plate1.flexural_rigidity)
print("AR", plate1.y_length / plate1.x_length)
print("MS", 2 * beam1.mass_per_unit_length / (plate1.mass_per_unit_area * plate1.y_length))
print("DP", plate1.flexural_rigidity)
print("m_p", plate1.mass_per_unit_area)
print("b", plate1.y_length)

print("BEAM FREQUENCIES: ", beam1.freqs)
print("PLATE FREQUENCIES: ", plate1.freqs)

constraints = [
    [np.pi / 4, np.pi / 2],
    [np.pi / 5, np.pi / 2],
    [2 * np.pi / 5, np.pi / 2],
    [np.pi / 3, np.pi / 2],
    [np.pi / 6, np.pi / 2],
]

plate1.constraint_eval = plate1.constraint_shapes(constraints)
beam1.constraint_eval = beam1.constraint_shapes(constraints)
plate1.constraint_shape_matrix = plate1.constraint_shapes(constraints)
dim = len(constraints)

roots = rayleigh_solve(
    beam1, plate1, constraints, 1000, 1e-9, 50, 1e-13, transform=True
)
print("ROOTS ARE: ", roots)

freq_range = np.linspace(0, 800, 7000)
dets = np.zeros((len(freq_range)))
eigenvals_real = []
trans = True

for indx, freq in enumerate(freq_range):
    A = construct_matrix_maybefast_tiny(freq, dim, beam1, plate1)
    C = 10 ** (-26)
    if trans == True:

        dets[indx] = sym_transform(np.linalg.det(A), C)
    else:
        dets[indx] = np.linalg.det(A)

plt.plot(freq_range, dets)

plt.show()

flag = True
print("SYSTEM INTERUPT")
print("[0] Find a root")
print("[1] Zoom In")
print("[X} Quit]")
while flag == True:
    raw = input("Select a number: ")
    if raw == "X":
        flag = False
    elif int(raw) == 0:
        lower = float(input("Lower"))
        upper = float(input("Upper"))
        root = bisection_method(
            lower, upper, 50, 1e-13, beam1, plate1, dim, transform=True
        )

        if root == None:
            print("No root")
        else:
            print(root)

        yes_no = input("Append root? [Y/N]")
        if yes_no == "Y":
            roots.append(root)

    elif int(raw) == 1:
        lower = float(input("Lower"))
        upper = float(input("Upper"))
        freq_range = np.linspace(lower, upper, 700)
        dets = np.zeros((len(freq_range)))
        eigenvals_real = []

        trans = True

        for indx, freq in enumerate(freq_range):
            A = construct_matrix_maybefast_tiny(freq, dim, beam1, plate1)
            C = 10 ** (-26)
            if trans == True:

                dets[indx] = sym_transform(np.linalg.det(A), C)
            else:
                dets[indx] = np.linalg.det(A)

        plt.plot(freq_range, dets)

        plt.show()


lambda_matrix = calc_lambda(roots, constraints, beam1, plate1)

qbar_plate = calc_qbar_plate(lambda_matrix, roots, plate1)


# out, X, Y = calc_modeshape_matrix(qbar_plate, 0, plate1)


for i in range(len(roots)):
    out, X, Y, constraint_z = calc_modeshape_matrix(
        qbar_plate, i, plate1, constraints
    )
    heatmap_plot(out, X, Y, plate1, constraints, constraint_z)

# heatmap_plot(X, Y, out)


freq_range = np.linspace(0, 5000, 7000)
dets = np.zeros((len(freq_range)))
eigenvals_real = []


trans = True

for indx, freq in enumerate(freq_range):
    A = construct_matrix_maybefast_tiny(freq, dim, beam1, plate1)
    C = 10 ** (-26)
    if trans == True:

        dets[indx] = sym_transform(np.linalg.det(A), C)
    else:
        dets[indx] = np.linalg.det(A)


plt.plot(freq_range, dets)

plt.show()

