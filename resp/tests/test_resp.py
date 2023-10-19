from __future__ import division, absolute_import, print_function

import pytest
import sys

import numpy as np
import psi4

sys.path.insert(1, '../')

import driver as resp


def test_resp_unconstrained_a():
    ''' One-stage fitting of charges.

        - ESP is compute using Psi4
        - resp_a = 0.001

        Reference charges were generated by R.E.D.-III.5 and GAMESS.
    '''

    reference_charges = np.array([-0.294974,  0.107114,  0.107114,  0.084795,
                                   0.803999, -0.661279,  0.453270, -0.600039])

    charges = resp.resp('resp_unconstrained_a.ini')

    print('Unrestrained Electrostatic Potential Charges')
    print(f'{charges[0]}\n')

    print('Restrained Electrostatic Potential (RESP) Charges')
    print(f'{charges[1]}\n')

    print('Reference RESP Charges (via RED-III 5)')
    print(f'{reference_charges}\n')

    print('Difference')
    print(f'{charges[1]-reference_charges}\n')

    assert np.allclose(charges[1], reference_charges, atol=5e-4)


def test_resp_unconstrained_b():
    ''' One-stage fitting of charges.

        - ESP is read in (i.e., output from previous test)
        - resp_a = 0.001

        Reference charges were generated by R.E.D.-III.5 and GAMESS.
    '''

    reference_charges = np.array([-0.294974,  0.107114,  0.107114,  0.084795,
                                   0.803999, -0.661279,  0.453270, -0.600039])

    charges = resp.resp('resp_unconstrained_b.ini')

    print('Unrestrained Electrostatic Potential Charges')
    print(f'{charges[0]}\n')

    print('Restrained Electrostatic Potential (RESP) Charges')
    print(f'{charges[1]}\n')

    print('Reference RESP Charges (via RED-III 5)')
    print(f'{reference_charges}\n')

    print('Difference')
    print(f'{charges[1]-reference_charges}\n')

    assert np.allclose(charges[1], reference_charges, atol=5e-4)


def test_resp_constrained_a():
    ''' Two-stage fitting of charges.

        - ESP is compute using Psi4
        - resp_a = 0.001
        - last four atoms constrained to certain values
        - methyl hydrogens force to be equivalent
    '''

    reference_charges = np.array([-0.290893,  0.098314,  0.098314,  0.098314,
                                   0.803999, -0.661279,  0.453270, -0.600039])

    charges = resp.resp('resp_constrained_a.ini')

    print('Unrestrained Electrostatic Potential Charges')
    print(f'{charges[0]}\n')

    print('Restrained Electrostatic Potential (RESP) Charges')
    print(f'{charges[1]}\n')

    print('Reference RESP Charges (via RED-III 5)')
    print(f'{reference_charges}\n')

    print('Difference')
    print(f'{charges[1]-reference_charges}\n')

    assert np.allclose(charges[1], reference_charges, atol=5e-4)


def test_resp_two_conformers_a():

#     # Initialize two different conformations of ethanol
#     geometry = """C    0.00000000  0.00000000  0.00000000
#     C    1.48805540 -0.00728176  0.39653260
#     O    2.04971655  1.37648153  0.25604810
#     H    3.06429978  1.37151670  0.52641124
#     H    1.58679428 -0.33618761  1.43102358
#     H    2.03441010 -0.68906454 -0.25521028
#     H   -0.40814044 -1.00553466  0.10208540
#     H   -0.54635470  0.68178278  0.65174288
#     H   -0.09873888  0.32890585 -1.03449097
#     """
#     mol1 = psi4.geometry(geometry)
#     mol1.update_geometry()
#     mol1.set_name('conformer1')

#     geometry = """C    0.00000000  0.00000000  0.00000000
#     C    1.48013500 -0.00724300  0.39442200
#     O    2.00696300  1.29224100  0.26232800
#     H    2.91547900  1.25572900  0.50972300
#     H    1.61500700 -0.32678000  1.45587700
#     H    2.07197500 -0.68695100 -0.26493400
#     H   -0.32500012  1.02293415 -0.30034094
#     H   -0.18892141 -0.68463906 -0.85893815
#     H   -0.64257065 -0.32709111  0.84987482
#     """
#     mol2 = psi4.geometry(geometry)
#     mol2.update_geometry()
#     mol2.set_name('conformer2')

#     molecules = [mol1, mol2]

#     # Specify options
#     options = {'VDW_SCALE_FACTORS' : [1.4, 1.6, 1.8, 2.0],
#                'VDW_POINT_DENSITY'  : 1.0,
#                'RESP_A'             : 0.0005,
#                'RESP_B'             : 0.1,
#                'RESTRAINT'          : True,
#                'IHFREE'             : False,
#                'WEIGHT'             : [1, 1],
#                }

#     # Call for first stage fit
#     charges1 = resp.resp(molecules, options)
    charges = resp.resp('resp_two_confs.ini')

    # print("Restrained Electrostatic Potential Charges")
    # print(charges1[1])
    # # Reference Charges are generates with the resp module of Ambertools
    # # Grid and ESP values are from this code with Psi4
    reference_charges = np.array([-0.149134, 0.274292, -0.630868,  0.377965, -0.011016,
                                  -0.009444, 0.058576,  0.044797,  0.044831])
    # print("Reference RESP Charges")
    # print(reference_charges1)
    # print("Difference")
    # print(charges1[1]-reference_charges1)

    print('Unrestrained Electrostatic Potential Charges')
    print(f'{charges[0]}\n')

    print('Restrained Electrostatic Potential (RESP) Charges')
    print(f'{charges[1]}\n')

    print('Reference RESP Charges (via RED-III 5)')
    print(f'{reference_charges}\n')

    print('Difference')
    print(f'{charges[1]-reference_charges}\n')

    assert np.allclose(charges[1], reference_charges, atol=1e-5)

    # # Add constraint for atoms fixed in second stage fit
    # options['resp_a'] = 0.001
    # resp.set_stage2_constraint(molecules[0], charges1[1], options)

    # options['grid'] = []
    # options['esp'] = []
    # for mol in range(len(molecules)):
    #     options['grid'].append('%i_%s_grid.dat' %(mol+1, molecules[mol].name()))
    #     options['esp'].append('%i_%s_grid_esp.dat' %(mol+1, molecules[mol].name()))

    # # Call for second stage fit
    # print(molecules)
    # print(options)
    # charges2 = resp.resp(molecules, options)
    # print("\nStage Two\n")
    # print("RESP Charges")
    # print(charges2[0][1])
    # reference_charges2 = np.array([-0.079853, 0.253918, -0.630868, 0.377965, -0.007711,
    #                                -0.007711, 0.031420,  0.031420, 0.031420])
    # print("Reference RESP Charges")
    # print(reference_charges2)
    # print("Difference")
    # print(charges2[1]-reference_charges2)

    # assert np.allclose(charges2[1], reference_charges2, atol=1e-5)
