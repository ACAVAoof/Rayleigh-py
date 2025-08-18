from calculateMatrix import construct_matrix_maybefast_tiny
import numpy as np


def rayleigh_solve(
    beam,
    plate,
    constraints,
    freq_range,
    eps,
    loops,
    stopping_eps,
    transform=False,
):

    ##get the sorted list of frequencies
    ##apply the epsilon factor
    ## [ [freq1-eps, freq1+eps], [freq2-eps, freq2+eps]

    freqs = beam.freqs
    freqs = np.append(freqs, plate.freqs)
    freqs.sort()

    mod_freqs = pad(freqs, eps)

    base_root_num = 1

    for indx in range(len(constraints)):
        dim = indx + 1
        roots = []
        mod_freqs_temp = []
        for indy in range(len(mod_freqs) - 1):

            if (
                len(roots) >= base_root_num and indx != 0
            ):  ##if we have found enough roots, break out
                break

            if (
                mod_freqs[indy + 1][0] > freq_range
            ):  ##if we are outside of the specified freq_range
                break

            beam.constraint_eval = beam.constraint_shapes(
                constraints[0 : indx + 1]
            )
            plate.constraint_eval = plate.constraint_shapes(
                constraints[0 : indx + 1]
            )

            # print(f'On s = {indx} case')
            root = bisection_method(
                mod_freqs[indy][1],
                mod_freqs[indy + 1][0],
                loops,
                stopping_eps,
                beam,
                plate,
                dim,
                transform,
            )

            if root == None:  ##if there wasn't a root
                continue
            else:  ##if a root was found
                roots.append(root)
                root = [root - eps, root + eps]
                mod_freqs_temp.append(root)  ##also sort this as well

        if (
            indx == 0
        ):  ##on the first loop, we know there will be a maximum of this many roots in the range
            base_root_num = len(roots)

        mod_freqs = mod_freqs + mod_freqs_temp
        mod_freqs.sort(
            key=lambda x: x[0]
        )  ##funky call to sort because this is list of lists

    return roots


def asym_solve(
    beam,
    plate,
    constraints,
    freq_range,
    eps,
    loops,
    stopping_eps,
    transform=False,
):

    freqs = beam.freqs
    freqs = np.append(freqs, plate.freqs)
    freqs.sort()

    unique_keys = []
    [unique_keys.append(val) for val in freqs if val not in unique_keys]

    mod_freqs = pad(unique_keys, eps)
    roots = []

    beam.constraint_eval = beam.constraint_shapes(constraints)
    plate.constraint_eval = plate.constraint_shapes(constraints)
    dim = len(constraints)

    for indy in range(len(mod_freqs) - 1):

        if (
            mod_freqs[indy + 1][0] > freq_range
        ):  ##if we are outside of the specified freq_range
            break

        root = bisection_method(
            mod_freqs[indy][1],
            mod_freqs[indy + 1][0],
            loops,
            stopping_eps,
            beam,
            plate,
            dim,
            transform,
        )

        if root == None:  ##if there wasn't a root
            continue
        else:  ##if a root was found
            roots.append(root)

    return roots


def bisection_method(
    lower_bound, upper_bound, loops, stopping_eps, beam, plate, dim, transform
):

    # print(f'Performing Bisection method between {lower_bound} and {upper_bound}')

    lower_det = np.linalg.det(
        construct_matrix_maybefast_tiny(lower_bound, dim, beam, plate)
    )
    upper_det = np.linalg.det(
        construct_matrix_maybefast_tiny(upper_bound, dim, beam, plate)
    )

    # print(f'Lower Det was {lower_det} and upper det was {upper_det}')

    C = (lower_det + upper_det) / 2

    if (lower_det > 0 and upper_det > 0) or (
        lower_det < 0 and upper_det < 0
    ):  ##same polarity
        # print(lower_bound, upper_bound, 'could not be estimated')
        return None

    if transform == True:
        lower_det = sym_transform(lower_det, C)
        upper_det = sym_transform(upper_det, C)

    num_loops = 0

    while num_loops < loops:

        middle = (lower_bound + upper_bound) / 2
        middle_det = np.linalg.det(
            construct_matrix_maybefast_tiny(middle, dim, beam, plate)
        )

        if transform == True:
            middle_det = sym_transform(middle_det, C)
        residual = abs(middle_det)

        if (
            residual < stopping_eps
        ):  ## if we are withing the stopping criterion
            # print(f'Reached stopping eps for root {middle}')
            # print(f'Number of loops was, {num_loops}')

            return middle

        if middle_det > 0 and lower_det < 0:
            upper_bound = middle
        elif middle_det < 0 and lower_det < 0:
            lower_bound = middle
        elif middle_det > 0 and lower_det > 0:
            lower_bound = middle
        elif middle_det < 0 and lower_det > 0:
            upper_bound = middle
        elif middle_det == 0:
            return middle
        elif lower_det == 0:
            return lower_bound
        elif upper_det == 0:
            return upper_bound
        else:
            print("I FREW UP!!!!!")
            print(lower_det, middle_det, upper_det)
            raise Exception("I FREQ UP!!!!")

        num_loops += 1

    ##if we weren't able to break out of the loop due to criterion
    # print(f'Stopping criterion was not reached for this root, {middle}.')
    # print(f'Number of loops was, {num_loops}')
    return middle


def pad(freqs, eps):
    mod_freq = []
    for freq in freqs:
        lower = freq - eps
        upper = freq + eps
        mod_freq.append([lower, upper])
    return mod_freq


def sym_transform(arg, C):
    return np.sign(arg) * (np.log10(1 + abs(arg / C)))
