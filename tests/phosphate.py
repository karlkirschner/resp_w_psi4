from __future__ import division, absolute_import, print_function

import pytest
import sys

import numpy as np
import psi4

#sys.path.insert(1, '/home/kkirsc3m/git/resp_w_psi4/src')
sys.path.insert(1, '/home/karl/git/resp_w_psi4/src')

import driver as resp

charges = resp.resp('phosphate.ini')

print('Unrestrained Electrostatic Potential Charges')
print(f'{charges[0]}\n')

print('Restrained Electrostatic Potential (RESP) Charges')
print(f'{charges[1]}\n')

