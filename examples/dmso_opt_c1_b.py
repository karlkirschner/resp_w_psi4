import os
import pytest
import sys

import numpy as np
import psi4

if not os.path.basename(os.getcwd()) == 'tests':
    pytest.exit("ERROR: This test suite must be run from within the /tests directory "
                "due to relative path dependencies for .ini and .xyz files.")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import driver as resp

#charges = resp.resp('dmso_opt_c1.ini')
charges = resp.resp('dmso_opt_c1_b.ini')

print('\nUnrestrained Electrostatic Potential Charges')
print(f'{charges[0]}\n')

print('Restrained Electrostatic Potential (RESP) Charges')
print(f'{charges[1]}\n')
