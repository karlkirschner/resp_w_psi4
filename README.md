Psi4-Compatible Restrained Electrostatic Potential
==================================================

This code is a refactoring of the Python RESP code developed by Alenaizan et al. (https://github.com/cdsgroup/resp), which is a plugin for the Psi4 quantum chemistry package.

The revised code extends the implementation of Alenaizan et al. by enabling partial atomic charge (PAC) fitting onto extra/dummy particles, and by introducing a standardized configuration workflow based on `.ini` inputs. This facilitates modeling molecules that exhibit strongly anisotropic electrostatics (e.g., $\sigma$-holes and electron lone pairs). It also supports straightforward reproduction of the workflow and the resulting PAC. The root mean square error (RMSE) and relative root mean square error (RRMSE) metrics are also introduced into the code and the resulting output file to assess the fit quality.

This version consolidates the workflow into an easy-to-follow pipeline:
- Molecular structures and fitting settings are read from external XYZ-formatted and `.ini` files.
- Grid and ESP generation are performed, or loaded, as before.
- PAC fitting is carried out using an updated multi-center formulation that includes both nuclei and extra/dummy points as charge sites.

## Dependencies

### Python standard library
- `configparser`
- `os`
- `copy`
- `sys`

### Third-party
- `numpy`
- `psi4` (https://psicode.org)
- `pytest` (optional; for running tests)

## Code
The following files are included:
- `resp/espfit.py`: Restrained electrostatic potential (RESP) fitting procedure
- `resp/driver.py`: Main driver.
- `resp/tests/test_resp.py`: PyTest script containing:
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
- [[Bayly:93:10269-10280](https://pubs.acs.org/doi/abs/10.1021/j100142a004)] Bayly C. I., Cieplak, P., Cornell, W., Kollman, P.A.,  *A well-behaved electrostatic potential based method using charge restraints for deriving atomic charges: the RESP model.* *J. Phys. Chem.* **97**, 10269 (1993).

Please cite this article if you use this program:
- [[Alenaizan:19](https://doi.org/10.1002/qua.26035)] Alenaizan A., Burns L. A., Sherrill C. D. *Python implementation of the restrained electrostatic potential charge model.* *Int. J. Quantum Chem.* **120**, e26035 (2020).
