"""
Fitting procedure for RESP charges.

Reference:
Equations taken from [Bayly:93:10269].
"""

from __future__ import division, absolute_import, print_function

import copy

import numpy as np


def esp_solve(A: np.ndarray, B: np.ndarray, warning_notes: list) -> np.ndarray:
    """ Solves for point charges: A*q = B.

        Args
            A : matrix A
            B : matrix B

        Returns
            q : charges

        Library dependencies
            numpy
            warnings
    """

    q = np.linalg.solve(A, B)

    # Warning for near singular matrix, in case np.linalg.solve doesn't detect a singularity
    if np.linalg.cond(A) > 1/np.finfo(A.dtype).eps:
        warning_notes.append("Warning: Possible fit problem in esp_solve function; singular matrix.")

    return q, warning_notes


def restraint(q: np.ndarray, A_unrestrained: np.ndarray,
              resp_a: float, resp_b: float,
              num_conformers: int,
              ihfree: bool, symbols: list) -> np.ndarray:
    """ Add a hyperbolic restraint to matrix A.

        Args
            q              : charges
            A_unrestrained : unrestrained A matrix
            resp_a         : restraint scale a
            resp_b         : restraint parabola tightness b
            ihfree         : if hydrogens are excluded or included in restraint
            symbols        : element symbols
            num_conformers : the number of conformers

        Returns
            a : restrained A array

        Library dependencies
            copy
            numpy

        References
            1. Bayly, C. I.; Cieplak, P.; Cornell, W. & Kollman, P. A. A well-behaved
                electrostatic potential based method using charge restraints for deriving
                atomic charges: the RESP model J. Phys. Chem., 1993, 97, 10269-10280
                (Eqs. 10, 13)
    """

    if not isinstance(q, np.ndarray):
        raise TypeError(f'The input charges is not given as a np.ndarray (i.e., {q} variable).')
    elif not isinstance(A_unrestrained, np.ndarray):
        raise TypeError(f'The unrestrained A matrix is not given as a np.ndarray (i.e., {A_unrestrained} variable).')
    elif not isinstance(resp_a, float):
        raise TypeError(f'The resp_a is not given as a float (i.e., {resp_a} variable).')
    elif not isinstance(resp_b, float):
        raise TypeError(f'The resp_b is not given as a float (i.e., {resp_b} variable).')
    elif not isinstance(ihfree, bool):
        raise TypeError(f'The ihfree is not given as a boolean (i.e., {ihfree} variable).')
    elif not isinstance(symbols, list):
        raise TypeError(f'The element symbols is not given as a list (i.e., {symbols} variable).')
    elif not isinstance(num_conformers, int):
        raise TypeError(f'The num_conformers is not given as a float (i.e., {num_conformers} variable).')
    else:
        A = copy.deepcopy(A_unrestrained)

        for i in range(len(symbols)):
            # if an element is not hydrogen or if hydrogens are to be restrained
            # hyperbolic restraint: reference 1 (Eqs. 10, 13)
            if not ihfree or symbols[i] != 'H':
                A[i, i] = A_unrestrained[i, i] + resp_a/np.sqrt(q[i]**2 + resp_b**2) * num_conformers

        return A


def iterate(q: np.ndarray, A_unrestrained: np.ndarray, B: np.ndarray,
            resp_a: float, resp_b: float, toler: float,
            max_it: int, num_conformers: int,
            ihfree: bool, symbols: list,
            warning_notes: list) -> np.ndarray:
    """ Iterates the RESP fitting procedure.

        Args:
            q              : initial charges
            A_unrestrained : unrestrained A matrix
            B              : matrix B
            resp_a         : restraint scale a
            resp_b         : restraint parabola tightness b
            toler          : tolerance for charges in the fitting
            max_it         : maximum iteration number
            num_conformers : number of conformers
            ihfree         : if hydrogens are excluded or included in restraint
            symbols        : element symbols
            warning_notes  : warnings that are generated

        Returns
            q : fitted charges

        Library dependencies
            copy
            numpy
    """

    if not isinstance(q, np.ndarray):
        raise TypeError(f'The input charges is not given as a np.ndarray (i.e., {q} variable).')
    elif not isinstance(A_unrestrained, np.ndarray):
        raise TypeError(f'The unrestrained A matrix is not given as a np.ndarray (i.e., {A_unrestrained} variable).')
    elif not isinstance(B, np.ndarray):
        raise TypeError(f'The B matrix is not given as a np.ndarray (i.e., {B} variable).')
    elif not isinstance(resp_a, float):
        raise TypeError(f'The resp_a is not given as a float (i.e., {resp_a} variable).')
    elif not isinstance(resp_b, float):
        raise TypeError(f'The resp_b is not given as a float (i.e., {resp_b} variable).')
    elif not isinstance(ihfree, bool):
        raise TypeError(f'The ihfree is not given as a boolean (i.e., {ihfree} variable).')
    elif not isinstance(symbols, list):
        raise TypeError(f'The element symbols is not given as a list (i.e., {symbols} variable).')  #  was np.ndarray
    elif not isinstance(toler, float):
        raise TypeError(f'The toler is not given as a float (i.e., {toler} variable).')
    elif not isinstance(max_it, int):
        raise TypeError(f'The max_it is not given as a float (i.e., {max_it} variable).')
    elif not isinstance(num_conformers, int):
        raise TypeError(f'The num_conformers is not given as a float (i.e., {num_conformers} variable).')
    elif not isinstance(warning_notes, list):
        raise TypeError(f'The warning_notes is not given as a float (i.e., {warning_notes} variable).')
    else:
        q_last = copy.deepcopy(q)

        n_it = 0  # current number of interation
        difference = 2*toler

        while (difference > toler) and (n_it < max_it):
            n_it += 1
            A = restraint(q=q, A_unrestrained=A_unrestrained,
                          resp_a=resp_a, resp_b=resp_b, num_conformers=num_conformers, ihfree=ihfree, symbols=symbols)

            q, warning_notes = esp_solve(A=A, B=B, warning_notes=warning_notes)

            # Extract vector elements that correspond to charges
            difference = np.sqrt(np.max((q[:len(symbols)] - q_last[:len(symbols)])**2))
            q_last = copy.deepcopy(q)

        if difference > toler:
            warning_notes.append(f"Warning: Charge fitting unconverged; try increasing max iteration number to >{max_it}.")

        return q[:len(symbols)], warning_notes


def intramolecular_constraints(constraint_charge: dict, equivalent_groups: list):
    """ Extracts intramolecular constraints from user constraint input

        Args
            constraint_charge : a list of lists of charges and atom indices list, 
                e.g., [[0, [1, 2]], [1, [3, 4]]].
                The sum of charges on 1 and 2 will equal 0.
                The sum of charges on 3 and 4 will equal 1.
            equivalent_groups : a list of lists of indices of atoms to have equal charge,
                e.g., [[1, 2], [3, 4]]
                atoms 1 and 2 will have equal charge
                atoms 3 and 4 will have equal charge

        Returns
            constrained_charges (list) : fixed charges
            constrained_indices (list) : list of lists of indices of atoms in a constraint;
                a negative number before an index means that atom's charge will be subtracted.

        Notes
            Atom indices starts with 1 not 0.
            Total charge constraint is added by default for the first molecule.
    """

    if not isinstance(constraint_charge, dict):
        raise TypeError(f'The input options are not a dictionary (i.e., {constraint_charge} variable).')
    elif not isinstance(equivalent_groups, list):
        raise TypeError(f'The input option is not a list (i.e., {equivalent_groups} variable).')
    else:
        constrained_charges = []
        constrained_indices = []

        # for i in constraint_charge:
        #     constrained_charges.append(i[0])
        #     group = []
        #     for k in i[1]:
        #         group.append(k)
        #     constrained_indices.append(group)

        for key, value in constraint_charge.items():
            constrained_charges.append(value)
            constrained_indices.append([key])

        for i in equivalent_groups:
            for j in range(1, len(i)):
                group = []
                constrained_charges.append(0)  # TODO: Assume the target value for each equivalent_groups is 0.0
                group.append(-i[j-1])
                group.append(i[j])
                constrained_indices.append(group)

        # print('TEST', constrained_charges, constrained_indices)
        return constrained_charges, constrained_indices


def fit(options: dict, data: dict):
    """ Performs ESP and RESP fits.

        Args
            options : fitting options and internal data

        Returns
            q_fitted (np.ndarray)  : fitted charges
            fitting_methods (list) : strings of fitting methods (i.e., ESP and RESP)
            warning_notes          : warnings concerning the fitting

        Library dependencies
            numpy

        References
            1. Bayly, C. I.; Cieplak, P.; Cornell, W. & Kollman, P. A. A well-behaved
                electrostatic potential based method using charge restraints for deriving
                atomic charges: the RESP model J. Phys. Chem., 1993, 97, 10269-10280
                (Eqs. 12-14)
    """

    if not isinstance(options, dict):
        raise TypeError(f'The input options are not a dictionary (i.e., {options} variable).')
    elif not isinstance(data, dict):
        raise TypeError(f'The input options are not a dictionary (i.e., {data} variable).')
    else:
        q_fitted = []
        fitting_methods = []

        if (options['constraint_charge'] != 'None') and (options['equivalent_groups'] != 'None'):
            constraint_charges, constraint_indices = intramolecular_constraints(constraint_charge=options['constraint_charge'],
                                                                                equivalent_groups=options['equivalent_groups'])
        else:
            constraint_charges = []
            constraint_indices = []

        natoms = data['natoms']
        ndim = natoms + len(constraint_charges) + 1

        A = np.zeros((ndim, ndim))
        B = np.zeros(ndim)

        # Reference [1] (Eqs. 12-14)
        for mol in range(len(data['inverse_dist'])):
            r_inverse, V = data['inverse_dist'][mol], data['esp_values'][mol]

            # The a_matrix and b_vector are for one molecule, without the addition of constraints.
            # Construct a_matrix: a_jk = sum_i [(1/r_ij)*(1/r_ik)] Eq. 12 -> i.e., 1/r^2
            a_matrix = np.einsum("ij, ik -> jk", r_inverse, r_inverse)

            # Construct b_vector: b_j = sum_i (V_i/r_ij) -> i.e., esp/r
            b_vector = np.einsum('i, ij->j', V, r_inverse)

            # Weight the molecule
            a_matrix *= options['weight'][mol]**2
            b_vector *= options['weight'][mol]**2

            A[:natoms, :natoms] += a_matrix  # for atoms only, replace their values
            B[:natoms] += b_vector

        # print("KNK weight",options['weight'])
        # print("KNK A", A)
        # print("KNK B", B)

        # Include total charge constraint
        A[:natoms, natoms] = 1  # insert 1 in column after atoms [row, column]
        A[natoms, :natoms] = 1
        B[natoms] = data['mol_charge']

        # print("KNK A", A)
        # print("KNK B", B)
        # print('KNK mol_charge', data['mol_charge'])
        # print('KNK natoms', natoms)
        print('KNK constraint_charges', constraint_charges)
        print('KNK constraint_indices', constraint_indices)

        # Include constraints to matrices A and B
        for i in range(len(constraint_charges)):
            B[natoms + 1 + i] = constraint_charges[i]

            for k in constraint_indices[i]:
                if k > 0:
                    A[natoms + 1 + i, k - 1] = 1
                    A[k - 1, natoms + 1 + i] = 1
                else:
                    A[natoms + 1 + i, -k - 1] = -1
                    A[-k - 1, natoms + 1 + i] = -1

        # ESP
        fitting_methods.append('esp')
        q, warning_notes = esp_solve(A=A, B=B, warning_notes=data['warnings'])
        q_fitted.append(q[:natoms])

        # RESP
        if options['restraint']:
            fitting_methods.append('resp')
            q, warning_notes = iterate(q=q, A_unrestrained=A, B=B,
                                       resp_a=options['resp_a'], resp_b=options['resp_b'], toler=options['toler'],
                                       max_it=options['max_it'], num_conformers=len(data['inverse_dist']),
                                       ihfree=options['ihfree'], symbols=data['symbols'],
                                       warning_notes=data['warnings'])
            q_fitted.append(q)

        data['fitted_charges'] = q_fitted
        data['fitting_methods'] = fitting_methods
        data['warnings'] = warning_notes

        return data
