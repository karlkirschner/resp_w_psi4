# Restrained Electrostatic Potential (RESP) Charges with Extra Point Fitting
> Compute RESP partial atomic charges (PACs) from Psi4 electrostatic potentials. This tool supports fitting to off-center points, such as electron lone-pair regions and sigma holes, for improved electrostatic modeling.

- Author: Karl N. Kirschner

---
## Overview

This tool is a refactoring of the Python RESP code developed by Alenaizan et al. (https://github.com/cdsgroup/resp), which is a plugin for the Psi4 quantum chemistry package. This revision extends the original code by enabling PAC fitting onto virtual sites (a.k.a., extra points, dummy particles), facilitating the modeling of molecules that exhibit strongly anisotropic electrostatics (e.g., $\sigma$-holes and electron lone pairs). Additionally, a standardized configuration workflow was introduced based on `.ini` inputs, enabling straightforward reproduction of workflows and their resulting PACs. The root mean square error (RMSE) and relative root mean square error (RRMSE) metrics are also included for fit quality assessment.

## Features
This version consolidates the workflow into an easy-to-follow pipeline:
- Molecular structures and fitting settings are read from external XYZ-formatted and `.ini` files.
- Grid and ESP generation are performed using Psi4, or can be loaded from a previous calculation.
- PAC fitting is performed using an updated multi-center formulation that includes both nuclei and extra points as charge sites.
- Multiple conformers can be used to generate an averaged PAC set.
- Charge equivalency and restraints can also be specified to specific atoms.
- RMSE and RRMSE metrics are computed.

## Dependencies

### Python standard library
- `configparser`
- `copy`
- `os`
- `sys`

### Third-party
- `numpy`
- `psi4` (https://psicode.org)
- `pytest` (optional; for running tests)

## Code
The following files are included:
- `resp/espfit.py`: Restrained electrostatic potential (RESP) fitting procedure and metrics
- `resp/driver.py`: Main driver
- `resp/tests/test_resp.py`: PyTest script containing different fitting scenarios:
  - `test_resp_unconstrained_a()`
  - `test_resp_unconstrained_b()`
  - `test_resp_constrained_a()`
  - `test_resp_two_conformers_a()`
  - `test_resp_two_conformers_b()`
  - `test_bromoethene()`
  - `test_bromoethene_x()` - (1 extra point)
  - `test_methanol()`
  - `test_methanol_x()` - (2 extra points)
- `resp/vdw_surface.py`: Van der Waals surface generation
- `resp/stage2_helper.py`: Helper utilities for two-stage fitting

## References
* [[Bayly:93:10269-10280](https://pubs.acs.org/doi/abs/10.1021/j100142a004)] Bayly C. I., Cieplak, P., Cornell, W., Kollman, P.A.,  *A well-behaved electrostatic potential based method using charge restraints for deriving atomic charges: the RESP model.* *J. Phys. Chem.* **97**, 10269 (1993).

Please cite the following articles if you use this program:
* [[Alenaizan:19](https://doi.org/10.1002/qua.26035)] Alenaizan A., Burns L. A., Sherrill C. D. *Python implementation of the restrained electrostatic potential charge model.* *Int. J. Quantum Chem.* **120**, e26035 (2020).
* Kirschner KN, manuscript in preparation (2026)
