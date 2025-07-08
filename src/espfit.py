"""
Fitting procedure for RESP charges.

Reference:
Equations taken from [Bayly:93:10269].
"""

import copy

import numpy as np


def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """ Calculates the Root Mean Square Error (RMSE) between predicted and target values.

        Args:
            predictions: predicted values
            targets    : target (observed|actual) values

        Returns:
            RMSE value
    """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    else:
        return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_rrms(rmse_value: float, targets: np.ndarray) -> float:
    """ Calculates the Relative Root Mean Square (RRMS) given an RMSE value and target values.

        Args:
            rmse_value: pre-calculated RMSE value
            targets:    target (observed|actual) values

        Returns:
            RRMS value (dimensionless)
    """
    rms_targets = np.sqrt(np.mean(targets ** 2)) # rms

    if rms_targets == 0:
        print("Warning: RMS of target values is zero, RRMS cannot be calculated.")
        return np.nan
    else:
        rrms = rmse_value / rms_targets # relative

    return rrms


def calculate_esp_metrics(q_fitted_esp: np.ndarray, data: dict) -> dict:
    """ Calculates the Root Mean Square Error (RMSE) and Relative Root Mean Square (RRMS)
        of the fitted charges against the original QM ESP values at the grid points.

        Args:
            q_fitted_esp (np.ndarray): The 1D array of fitted partial atomic charges (for ESP).
                                       This should correspond to q[:natoms] from the fit function.
            data (dict): A dictionary containing necessary data, specifically:
                - 'inverse_dist' (list of np.ndarray): List of inverse distance matrices (1/r_ij)
                                                       for each conformer/molecule.
                - 'esp_values' (list of np.ndarray): List of original quantum mechanical ESP values
                                                     at grid points for each conformer/molecule.

        Returns:
            dict: A dictionary containing 'true_esp_rmse' and 'true_esp_rrms'.
    """
    all_recalculated_esp_values = []
    all_original_esp_values = []

    num_conformers = len(data['inverse_dist'])

    for mol_idx in range(num_conformers):
        # The inverse_dist matrix here is effectively 1/r_ij where i is grid point, j is atom
        inv_r_matrix_for_conformer = data['inverse_dist'][mol_idx]

        # Original QM ESP values for this conformer
        original_V_qm_conformer = data['esp_values'][mol_idx]

        # Recalculate ESP at grid points using fitted charges
        # V_calc_i = sum_j (q_j / r_ij)
        # This is a matrix-vector product: (num_esp_points x num_atoms) @ (num_atoms x 1)
        recalculated_V_esp_conformer = np.dot(inv_r_matrix_for_conformer, q_fitted_esp)

        all_recalculated_esp_values.extend(recalculated_V_esp_conformer)
        all_original_esp_values.extend(original_V_qm_conformer)

    # lists to arrays
    all_recalculated_esp_values = np.array(all_recalculated_esp_values)
    all_original_esp_values = np.array(all_original_esp_values)

    print(f'\nNumber of predicted values along grid points: {len(all_recalculated_esp_values)}')
    print(f'Number of target values along grid points: {len(all_original_esp_values)}\n')

    true_esp_rmse = calculate_rmse(predictions=all_recalculated_esp_values, targets=all_original_esp_values)
    true_esp_rrms = calculate_rrms(rmse_value=true_esp_rmse, targets=all_original_esp_values)

    return {'true_esp_rmse': true_esp_rmse,
            'true_esp_rrms': true_esp_rrms}


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
        raise TypeError(f'The num_conformers is not given as a int (i.e., {num_conformers} variable).')
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
            B              : matrix B (i.e., the QM target values)
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
        raise TypeError(f'The element symbols is not given as a list (i.e., {symbols} variable).')
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

        n_it = 0
        difference = 2*toler

        while (difference > toler) and (n_it < max_it):
            n_it += 1
            A = restraint(q=q, A_unrestrained=A_unrestrained,
                          resp_a=resp_a, resp_b=resp_b, num_conformers=num_conformers, ihfree=ihfree, symbols=symbols)

            q, warning_notes = esp_solve(A=A, B=B, warning_notes=warning_notes)

            # Convergence check
            # Extract vector elements that correspond to charges
            difference = np.sqrt(np.max((q[:len(symbols)] - q_last[:len(symbols)])**2))
            q_last = copy.deepcopy(q)

        if difference > toler:
            warning_notes.append(f"Warning: Charge fitting unconverged; try increasing max iteration number to >{max_it}.")

        return q[:len(symbols)], warning_notes


def intramolecular_constraints(constraint_charge: (dict, bool), equivalent_groups: (list, bool)):
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
    if not isinstance(constraint_charge, (dict, bool)):
        raise TypeError(f'The input options are not a dictionary or boolean (i.e., {constraint_charge} variable).')
    elif not isinstance(equivalent_groups, (list, bool)):
        raise TypeError(f'The input option is not a list or boolean (i.e., {equivalent_groups} variable).')
    else:
        constrained_charges = []
        constrained_indices = []

        if constraint_charge:
            for key, value in constraint_charge.items():
                constrained_charges.append(value)
                constrained_indices.append([key])
        if equivalent_groups:
            for i in equivalent_groups:
                for j in range(1, len(i)):
                    group = []
                    constrained_charges.append(0.0)  # Target value for equivalent charge constraints is zero (q_A - q_B = 0)
                    group.append(-i[j-1])
                    group.append(i[j])
                    constrained_indices.append(group)

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
        if (options['constraint_charge'] is not None) and (options['equivalent_groups'] is not None):
            constraint_charges, constraint_indices = intramolecular_constraints(constraint_charge=options['constraint_charge'],
                                                                                equivalent_groups=options['equivalent_groups'])
        elif (options['constraint_charge'] is None) and (options['equivalent_groups'] is not None):
            constraint_charges, constraint_indices = intramolecular_constraints(constraint_charge=False,
                                                                                equivalent_groups=options['equivalent_groups'])
        elif (options['constraint_charge'] is not None) and (options['equivalent_groups'] is None):
            constraint_charges, constraint_indices = intramolecular_constraints(constraint_charge=options['constraint_charge'],
                                                                                equivalent_groups=False)
        else:
            constraint_charges = []
            constraint_indices = []

        natoms = data['natoms']
        ndim = natoms + len(constraint_charges) + 1

        # A: the matrix of coefficients in the linear system of equations (A mathbf{q} = B) that is solved to determine the partial atomic charges mathbf{q}.
        # mathbf{q} = vector of unknowns (i.e. the partial atomic charges)
        # B: target vector that consists of known values
        A = np.zeros((ndim, ndim))
        B = np.zeros(ndim)

        # Reference [1] (Eqs. 12-14)
        for mol in range(len(data['inverse_dist'])):
            # V: QM ESP values computed on grid points around a conformer/molecule
            r_inverse, V = data['inverse_dist'][mol], data['esp_values'][mol]

            # The a_matrix and b_vector are for one conformer/molecule, without the addition of constraints.
            # a_matrix: essentially a sum over all ESP grid points (i) of the product of (1/r_ij) * (1/r_ik)
            # Construct a_matrix: a_jk = sum_i [(1/r_ij)*(1/r_ik)] Eq. 12 -> i.e., 1/r^2
            a_matrix = np.einsum("ij, ik -> jk", r_inverse, r_inverse)

            # Construct b_vector: b_j = sum_i (V_i/r_ij) -> i.e., esp/r
            # b_vector: essentially a sum over all grid points (i) for each atom (j) of V_i / r_ij. This term directly relates the target ESP at the grid points to the atomic centers.
            b_vector = np.einsum('i, ij->j', V, r_inverse)

            # Weight the conformer/molecule and go ahead and square them eventual multiplying with least-square fit formula
            a_matrix *= options['weight'][mol]**2
            b_vector *= options['weight'][mol]**2

            # print(f"SHAPE A: {np.shape(A[:natoms, :natoms])}; a: {np.shape(a_matrix)}")
            # print(f"SHAPE B: {np.shape(B)}; b: {np.shape(b_vector)}")
            A[:natoms, :natoms] += a_matrix  # for atoms only, replace their values
            B[:natoms] += b_vector

        # Include total charge constraint
        A[:natoms, natoms] = 1  # insert 1 in column after atoms [row, column]
        A[natoms, :natoms] = 1
        B[natoms] = data['formal_charge']

        # print(f"SHAPE A final: {np.shape(A[:natoms, :natoms])}; a: {np.shape(a_matrix)}")
        # print(f"SHAPE B final: {np.shape(B)}; b: {np.shape(b_vector)}")

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
        q_fitted_esp = q[:natoms]
        q_fitted.append(q_fitted_esp)

        print(f"\nESP shapes - A:{np.shape(A)}; B:{np.shape(B)}; q: {np.shape(q)}")

        predictions = np.dot(A, q)  # original form: A*q

        # RMSE internal consistency check of the prediction's linear algebra
        rmse = calculate_rmse(predictions=predictions, targets=B)
        print(f"Internal ESP RMSE: {rmse}")

        # real fitting error: original QM ESP values and the ESP values recalculated from your fitted charges at the original grid points
        esp_metrics = calculate_esp_metrics(q_fitted_esp=q_fitted_esp, data=data)
        print(f"True ESP RMSE (vs original grid points): {esp_metrics['true_esp_rmse']}")
        print(f"True ESP RRMS (vs original grid points): {esp_metrics['true_esp_rrms']}\n")

        data['esp_rmse_true'] = esp_metrics['true_esp_rmse'] # Store RESP metrics
        data['esp_rrms_true'] = esp_metrics['true_esp_rrms'] # Store RESP metrics

        # RESP
        if options['restraint']:
            fitting_methods.append('resp')
            q, warning_notes = iterate(q=q, A_unrestrained=A, B=B,
                                       resp_a=options['resp_a'], resp_b=options['resp_b'], toler=options['toler'],
                                       max_it=options['max_it'], num_conformers=len(data['inverse_dist']),
                                       ihfree=options['ihfree'], symbols=data['symbols'],
                                       warning_notes=data['warnings'])
            q_fitted_esp = q[:natoms]
            q_fitted.append(q_fitted_esp)

            resp_metrics = calculate_esp_metrics(q_fitted_esp=q_fitted_esp, data=data)
            print(f"True RESP RMSE (vs original grid points): {resp_metrics['true_esp_rmse']}")
            print(f"True RESP RRMS (vs original grid points): {resp_metrics['true_esp_rrms']}")

            data['resp_rmse_true'] = resp_metrics['true_esp_rmse'] # Store RESP metrics
            data['resp_rrms_true'] = resp_metrics['true_esp_rrms'] # Store RESP metrics

        data['fitted_charges'] = q_fitted
        data['fitting_methods'] = fitting_methods
        data['warnings'] = warning_notes

        return data
