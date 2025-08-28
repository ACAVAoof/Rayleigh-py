#After having found frequencies for determinant to be 0, take frequency and put back into problem to calculate mode shape using lambda constraint matrix
#From lambda matrix, calculate q-bar, then mode shape vectors
#Also has plotting features

import numpy as np
from calculateMatrix import construct_matrix_maybefast_tiny
import matplotlib.pyplot as plt
from matplotlib import cm


def calc_lambda(roots, constraints, beam, plate):

    lambda_matrix = np.zeros((len(constraints), len(roots)))
    lambda_matrix[0, :] = 1

    for indx, root in enumerate(roots):
        matrix = construct_matrix_maybefast_tiny(
            root, len(constraints), beam, plate
        )
        print(f"matrix @ root {root}", matrix)
        sub_matrix = matrix[1:, 1:]
        sub_vec = -matrix[1:, 0]
        lamb_vector = np.linalg.solve(sub_matrix, sub_vec)
        lambda_matrix[1:, indx] = lamb_vector

    return lambda_matrix


def calc_qbar_plate(lambda_matrix, roots, plate):

    matrix = np.zeros((plate.x_indx, plate.y_indx, len(roots)))

    for indx, root in enumerate(roots):

        lamb_vec = lambda_matrix[:, indx]
        top = 0
        bottom = 0

        for rx in range(plate.x_indx):
            for ry in range(plate.y_indx):
                top = -np.dot(
                    lamb_vec, plate.constraint_shape_matrix[rx, ry, :]
                )
                bottom = plate.gen_mass * (
                    -(root**2) + plate.freqs[rx, ry] ** 2
                )
                matrix[rx, ry, indx] = top / bottom

    return matrix


def calc_modeshape_matrix(
    q_bar_plate_matrix, root_indx, plate, constraints, DIVISIONS=50
):

    x_space = np.linspace(0, plate.a, DIVISIONS)
    y_space = np.linspace(0, plate.b, DIVISIONS)

    X, Y = np.meshgrid(x_space, y_space, indexing="ij")

    out = np.zeros((len(x_space), len(y_space)))

    for inx, x in enumerate(x_space):
        for iny, y in enumerate(y_space):
            my_sum = 0
            for rx in range(plate.x_indx):
                for ry in range(plate.y_indx):
                    my_sum += q_bar_plate_matrix[
                        rx, ry, root_indx
                    ] * plate.shape(rx + 1, ry + 1, [x, y])

            out[inx, iny] = my_sum

    constraint_z = np.zeros((len(constraints)))
    for cindx, constraint in enumerate(constraints):
        my_sum = 0
        for rx in range(plate.x_indx):
            for ry in range(plate.y_indx):
                my_sum += q_bar_plate_matrix[rx, ry, root_indx] * plate.shape(
                    rx + 1, ry + 1, constraint
                )
        constraint_z[cindx] = my_sum
    return out, X, Y, constraint_z


def heatmap_plot(out, X, Y, plate, constraints, constraint_z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    c = ax.plot_surface(X, Y, out, cmap=cm.coolwarm)
    ax.set_box_aspect((plate.a, plate.b, 1))
    for cindx, constraint in enumerate(constraints):
        ##calculate the deflection
        ax.scatter(
            constraint[0],
            constraint[1],
            constraint_z[cindx] - 0.5,
            color="black",
            s=35,
            marker="o",
        )
    plt.show()
