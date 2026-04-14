"""
Driver for the RESP code.
"""

# Original Work:
__authors__ = "Asem Alenaizan"
__credits__ = ["Asem Alenaizan"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-04-28"

# Refactoring:
__authors__ = ["Karl N. Kirschner"]
__credits__ = ["Karl N. Kirschner"]
__date__ = "2026-04"

import configparser
import os

import numpy as np
import psi4

import espfit
import vdw_surface

bohr_to_angstrom = 0.52917721092


def write_results(flags_dict: dict, data: dict, output_file: str):
    """ Write fitting inputs, settings, and RESP/ESP charges to an output file.

        Args
            flags_dict
                Parsed user options controlling the workflow (e.g., VDW surface
                parameters, ESP generation inputs, restraint settings, and constraints).
                This is typically the dictionary returned by `parse_ini`.
            data
                Dictionary containing per-molecule/per-conformer data and fit results.
                Expected keys include, at minimum:
                    - 'vdw_radii' : dict[str, float]
                    - 'name' : list[str]
                    - 'symbols' : list[str]
                    - 'esp_values' : list[np.ndarray]  (used for grid point counts)
                    - 'warnings' : list[str]
                    - 'fitting_methods' : list[str] or iterable containing 'esp'/'resp'
                    - 'fitted_charges' : list[np.ndarray] (one array per method)
                Optional keys for statistics:
                    - 'esp_rmse_grid', 'resp_rmse_grid', 'esp_rrms_grid', 'resp_rrms_grid'
            output_file
                Path to the output report file to be (over)written.
            Return
                Output text file.
    """
    if not isinstance(flags_dict, dict):
        raise TypeError(f'The input flags were not given as a dictionary (i.e., {flags_dict}).')
    elif not isinstance(data, dict):
        raise TypeError(f'The resulting data were not given as a dictionary (i.e., {data}).')
    else:
        with open(output_file, 'w') as outfile:
            outfile.write("Electrostatic potential parameters\n")

            outfile.write(f"{' ':4s}van der Waals radii (Angstrom):\n")
            for element, radius in data['vdw_radii'].items():
                outfile.write(f"{' ':38s}{element} = {radius:.3f}\n")

            # --- VDW Scale Factors ---
            vdw_scale_factors_str = "None"
            if flags_dict.get('vdw_scale_factors') is not None:
                vdw_scale_factors_str = ' '.join([str(i) for i in flags_dict['vdw_scale_factors']])
            outfile.write(f"{' ':4s}VDW scale factors:{' ':16s}{vdw_scale_factors_str}\n")

            # --- VDW Point Density ---
            vdw_point_density_val = flags_dict.get('vdw_point_density', "None")
            outfile.write(f"{' ':4s}VDW point density:{' ':16s}{vdw_point_density_val}\n")

            # --- ESP Method ---
            method_esp_val = flags_dict.get('method_esp', "None")
            outfile.write(f"{' ':4s}ESP method:{' ':23s}{method_esp_val}\n")

            # --- ESP Basis Set ---
            if flags_dict.get('basis_esp') is None:
                outfile.write(f"{' ':4s}ESP basis set:{' ':20s}None\n")
            else:
                outfile.write(f"{' ':4s}ESP basis set:\n")
                for basis in flags_dict['basis_esp']:
                    outfile.write(f"{' ':38s}{basis}\n")

            outfile.write(f'\nGrid information\n')
            outfile.write(f"{' ':4s}Quantum ESP File(s):\n")
            # --- Quantum ESP Files ---
            if flags_dict.get('esp') is None: # Use .get()
                for conf_n in range(len(flags_dict['input_files'])):
                    outfile.write(f"{' ':38s}{data['name'][conf_n]}_grid_esp.dat\n")
            else:
                for conf_n in range(len(flags_dict['input_files'])):
                    outfile.write(f"{' ':38s}{flags_dict['esp'][conf_n]}\n")

            outfile.write(f"\n{' ':4s}Grid Points File(s) (# of points):\n")
            # --- Grid Points Files ---
            if flags_dict.get('grid') is None: # Use .get()
                for conf_n in range(len(flags_dict['input_files'])):
                    outfile.write(f"{' ':38s}{data['name'][conf_n]}_grid.dat ({len(data['esp_values'][conf_n])})\n")
            else:
                for conf_n in range(len(flags_dict['input_files'])):
                    outfile.write(f"{' ':38s}{flags_dict['grid'][conf_n]} ({len(data['esp_values'][conf_n])})\n")

            outfile.write('\nConstraints\n')
            # --- Charge Constraints ---
            if flags_dict.get('constraint_charge') is not None and flags_dict['constraint_charge']: # Check for None AND empty dict
                outfile.write(f"{' ':4s}Charge constraints:\n")
                for key, value in flags_dict['constraint_charge'].items():
                    outfile.write(f"{' ':38s}Atom {key} = {value}\n")
            else:
                outfile.write(f"{' ':4s}Charge constraints: None\n")

            # --- Equivalent Groups ---
            if flags_dict.get('equivalent_groups') is not None and flags_dict['equivalent_groups']: # Check for None AND empty list
                outfile.write(f"\n{' ':4s}Equivalent charges on atoms (group = atom numbers):\n")
                count = 1
                for i in flags_dict['equivalent_groups']:
                    outfile.write(f"{' ':38s}group_{count} = ")
                    outfile.write(' '.join(map(str, i)))
                    count += 1
                    outfile.write('\n')
            else:
                outfile.write(f"\n{' ':4s}Equivalent charges on atoms: None\n")

            outfile.write('\nRestraint\n')
            outfile.write(f"{' ':4s}ihfree:{' ':27s}{flags_dict['ihfree']:}\n")
            outfile.write(f"{' ':4s}resp_a:{' ':27s}{flags_dict['resp_a']:.4f}\n")
            outfile.write(f"{' ':4s}resp_b:{' ':27s}{flags_dict['resp_b']:.4f}\n")

            outfile.write('\nFit\n')
            if len(data['warnings']) > 0:
                outfile.write(f"{' ':4s}WARNINGS:\n")
                for i in data['warnings']:
                    outfile.write(f"{' ':8s}{i}\n")
                outfile.write("\n")

            outfile.write(f"{' ':4s}Electrostatic Potential Charges:\n")
            outfile.write(f"{' ':8s}Center  Symbol{' ':8s}")

            # Prepare fitting method headers
            method_headers = []
            if 'esp' in data['fitting_methods']:
                method_headers.append('ESP')
            if 'resp' in data['fitting_methods']:
                method_headers.append('RESP')

            for header in method_headers:
                outfile.write(f'{header:12s}')
            outfile.write('\n')

            # Write charges for each method
            for i in range(len(data['symbols'])):
                outfile.write(f"{' ':8s}{i + 1:3d}{' ':8s}{data['symbols'][i]:2s}")
                # Assuming data['fitted_charges'] is a list where index 0 is ESP, index 1 is RESP
                if 'esp' in data['fitting_methods']:
                    outfile.write(f"{' ':4s}{data['fitted_charges'][0][i]:12.8f}")
                if 'resp' in data['fitting_methods'] and len(data['fitted_charges']) > 1:
                    outfile.write(f"{data['fitted_charges'][1][i]:12.8f}")
                outfile.write('\n')

            outfile.write(f"\n{' ':8s}Total Charge:{' ':4s}")
            for charges_set in data['fitted_charges']: # Iterate through the list of charge sets
                outfile.write(f'{np.sum(charges_set):12.8f}')
            outfile.write('\n')

            outfile.write(f"\n{' ':4s}Fitting Statistics:\n")
            outfile.write(f"{' ':8s}{'Metric':<10s}") # Left-align 'Metric'
            
            # headers (ESP, RESP)
            if 'esp' in data['fitting_methods']:
                outfile.write(f"{'ESP':>12s}") # Right-align ESP
            if 'resp' in data['fitting_methods']:
                outfile.write(f"{'RESP':>12s}") # Right-align RESP
            outfile.write('\n')

            # RMSE row
            outfile.write(f"{' ':8s}{'RMSE':<10s}")
            if 'esp' in data['fitting_methods'] and 'esp_rmse_grid' in data:
                outfile.write(f"{data['esp_rmse_grid']:12.5f}")
            else:
                outfile.write(f"{'N/A':>12s}") # If ESP RMSE not available
            
            if 'resp' in data['fitting_methods'] and 'resp_rmse_grid' in data:
                outfile.write(f"{data['resp_rmse_grid']:12.5f}")
            else:
                outfile.write(f"{'N/A':>12s}") # If RESP RMSE not available
            outfile.write('\n')

            # RRMS row
            outfile.write(f"{' ':8s}{'RRMS':<10s}")
            if 'esp' in data['fitting_methods'] and 'esp_rrms_grid' in data:
                outfile.write(f"{data['esp_rrms_grid']:12.5f}")
            else:
                outfile.write(f"{'N/A':>12s}") # If ESP RRMS not available

            if 'resp' in data['fitting_methods'] and 'resp_rrms_grid' in data:
                outfile.write(f"{data['resp_rrms_grid']:12.5f}")
            else:
                outfile.write(f"{'N/A':>12s}") # If RESP RRMS not available
            outfile.write('\n')


def parse_ini(input_ini: str) -> dict:
    """ Parse a configuration INI. The returned dictionary is suitable for direct use
        as the `flags_dict`/options argument throughout the workflow.

        Parsing rules
            - The literal string "None" (case-insensitive) is treated specially:
                * most keys are set to `None`
                * for keys that are naturally mapping-like (`vdw_radii`,
                  `constraint_charge`), an empty dict `{}` is returned to simplify
                  downstream iteration.
            - `input_files`: comma-separated list of XYZ paths (str).
            - `constraint_charge`: comma-separated "atom=value" pairs -> `{int: float}`.
            - `equivalent_groups`: comma-separated "group= i j k" entries -> `list[list[int]]`
              (group labels are ignored; only the atom-number lists are kept).
            - `esp`, `grid`, `basis_esp`: comma-separated lists of strings; empty lists
              are normalized to `None`.
            - `vdw_scale_factors`, `weight`: comma-separated lists of floats.
            - `vdw_radii`: comma-separated "Element=radius" pairs -> `{str: float}`.
            - `vdw_point_density`, `resp_a`, `resp_b`, `toler`: float.
            - `max_it`, `formal_charge`, `multiplicity`: int.
            - `restraint`, `ihfree`: boolean, parsed from "true"/"false" (case-insensitive).
            - All other keys are returned as raw strings.

        Args
            input_ini
                Path to an INI configuration file.

        Returns
        dict
            Parsed options with type conversions applied.
    """
    if not isinstance(input_ini, str):
        raise TypeError(f'The input_ini must be a string (i.e., {input_ini}).')

    config = configparser.ConfigParser()
    config.read(input_ini)
    flags_dict = {}

    def convert_none(value_str):
        if value_str.strip().lower() == 'none':
            return None
        return value_str

    for section in config.sections():
        for key in config[section]:
            raw_value = config.get(section, key)
            processed_value = convert_none(raw_value)

            # Handle explicit 'None' values
            if processed_value is None:
                # To make 'if x is None' checks in the driver work, 
                # we return None for almost everything.
                if key in ['vdw_radii', 'constraint_charge']:
                    flags_dict[key] = {} # Dicts are usually safe as {}
                else:
                    flags_dict[key] = None
                continue 

            # Type-specific parsing
            if key == 'input_files':
                flags_dict[key] = [item.strip().replace('\n', '') for item in processed_value.split(',')]

            elif key == 'constraint_charge':
                parsed_constraints = {}
                for constraint_str in processed_value.split(','):
                    if '=' in constraint_str:
                        atom_num, val = constraint_str.split('=')
                        parsed_constraints[int(atom_num.strip())] = float(val.strip())
                flags_dict[key] = parsed_constraints

            elif key == 'equivalent_groups':
                all_groups = []
                for group_str in processed_value.split(','):
                    if '=' in group_str:
                        atoms = group_str.split('=')[1].split()
                        all_groups.append([int(a) for a in atoms])
                flags_dict[key] = all_groups

            elif key in ['esp', 'grid', 'basis_esp']:
                items = [item.strip() for item in processed_value.split(',') if item.strip()]
                # If there's no real data, set to None so driver doesn't try to index it
                flags_dict[key] = items if items else None

            elif key in ['vdw_scale_factors', 'weight']:
                flags_dict[key] = [float(i.strip()) for i in processed_value.split(',')]

            elif key == 'vdw_radii':
                parsed_radii = {}
                for item_str in processed_value.split(','):
                    if '=' in item_str:
                        el, rad = item_str.split('=')
                        parsed_radii[el.strip()] = float(rad.strip())
                flags_dict[key] = parsed_radii

            elif key in ['vdw_point_density', 'resp_a', 'resp_b', 'toler']:
                flags_dict[key] = float(processed_value)

            elif key in ['max_it', 'formal_charge', 'multiplicity']:
                flags_dict[key] = int(processed_value)

            elif key in ['restraint', 'ihfree']:
                flags_dict[key] = processed_value.lower() == 'true'

            else:
                flags_dict[key] = processed_value

    return flags_dict


def read_xyz(infile: str, data_dict: dict) -> dict:
    """ Read an XYZ file, append basic per-geometry data to `data_dict`, and
        construct a corresponding Psi4 `Molecule`.

        Extract the molecule name from the XYZ filename stem, parses element
        symbols and Cartesian coordinates (Angstroms). A Psi4 molecule object is
        generated from the geometry with `nocom` and `noreorient` enabled to
        preserve the input coordinate frame.

        Args
            infile:
                path to an XYZ-formatted coordinate file.
            data_dict:
                used to accumulate per-geometry data across conformers. Must contain
                a list-valued key `name` prior to calling; this function appends
                to `name` and `coordinates`, and overwrites `symbols` and `natoms`.

        Returns
            data_dict:
                The updated data dictionary containing at least `name`, `symbols`,
                `natoms`, and `coordinates`.
            molecule:
                A Psi4 `psi4.core.Molecule` instance constructed from the XYZ geometry.

        Dependencies
            psi4
    """
    if not isinstance(infile, str):
        raise TypeError(f'The input XYZ-formatted file was not given as a string (i.e., {infile}).')
    elif not isinstance(data_dict, dict):
        raise TypeError(f'The data was not given as a dictionary (i.e., {data_dict}).')
    else:
        data_dict['symbols'] = []
        coordinates = []
        data_for_psi4 = []

        molec_name = os.path.splitext(infile)[0]
        data_dict['name'].append(molec_name)

        with open(infile) as input_file:
            next(input_file)
            next(input_file)
            for element_coord in input_file:
                data_for_psi4.append(element_coord)
                line = element_coord.strip().split(" ")
                line = list(filter(None, line))
                data_dict['symbols'].append(line[0].upper())
                coordinates.append(line[1:])

        data_for_psi4 = ''.join(line for line in data_for_psi4)  # Allows for additional commands (e.g. nocom)
        data_for_psi4 = "nocom\nnoreorient\n" + data_for_psi4  # Additional commands

        data_dict['natoms'] = len(data_dict['symbols'])

        coordinates = np.float64(coordinates)
        print(f'INPUT COORDS, ANGSTROMS\n {coordinates}')

        molecule = psi4.core.Molecule.from_string(data_for_psi4, name=molec_name)  # note: will returns coords in Bohr

        data_dict['coordinates'].append(coordinates)

        # print(f"FINAL FUNC DATA\n {data_dict['symbols']}\n {data_dict['natoms']}\n {type(coordinates)}\n {coordinates}\n\n")

        return data_dict, molecule


def resp(input_ini) -> list:
    """ RESP code driver.

    Args
        input_ini : input configuration for the calculation

    Returns
        charges : charges

    Notes
        output files : mol_results.dat: fitting results
                       mol_grid.dat: grid points in molecule.units
                       mol_grid_esp.dat: QM esp values in a.u.
    """
    if not isinstance(input_ini, str):
        raise TypeError(f'The input configparser file (i.e., {input_ini}) is not a str.')
    else:
        flags_dict = parse_ini(input_ini)

        if flags_dict.get('basis_esp') and flags_dict.get('esp'):
            raise ValueError("Error: both a basis set(s) and an input file(s) are specified for the ESP - choose one.")

        print('\nDetermining Partial Atomic Charges\n')

        output_file = input_ini.replace('ini', 'out')

        data = {}
        data['coordinates'] = []
        data['esp_values'] = []
        data['inverse_dist'] = []
        data['name'] = []
        data['warnings'] = []
        data['fitted_charges'] = []

        for conf_n in range(len(flags_dict['input_files'])):

            file_basename = flags_dict['input_files'][conf_n].replace('.xyz', '')

            data, conf = read_xyz(infile=flags_dict['input_files'][conf_n], data_dict=data)

            vdw_radii = {}  # units: Angstrom
            for element in data['symbols']:
                if element in flags_dict['vdw_radii']:
                    vdw_radii[element] = flags_dict['vdw_radii'][element]
                else:
                    # use built-in vdw_radii
                    vdw_radii[element] = vdw_surface.vdw_radii(element=element)

            data['vdw_radii'] = vdw_radii

            points = []  # units: Bohr

            if flags_dict.get('grid'):
                points = np.loadtxt(flags_dict['grid'][conf_n])

                if 'Bohr' in str(conf.units()):
                    points *= bohr_to_angstrom
            else:
                for scale_factor in flags_dict['vdw_scale_factors']:
                    computed_points = vdw_surface.vdw_surface(coordinates=data['coordinates'][conf_n],
                                                              element_list=data['symbols'],
                                                              scale_factor=scale_factor,
                                                              density=flags_dict['vdw_point_density'],
                                                              radii=data['vdw_radii'])
                    points.append(computed_points)

                points = np.concatenate(points)

                if 'Bohr' in str(conf.units()):
                    points /= bohr_to_angstrom
                    np.savetxt('grid.dat', points, fmt='%15.10f')  # units: Angstroms
                    points *= bohr_to_angstrom
                else:
                    np.savetxt('grid.dat', points, fmt='%15.10f')

            # Calculate ESP values along the grid points.
            # From Manual, v. 1.10a1.dev86":
            #    "The grid.dat file is completely free form; any number of spaces and/or
            #     newlines between entries is permitted. The units of the coordinates in
            #     grid.dat are the same as those used to specify the molecule’s geometry,
            #     and the output quantities are always in atomic units."
            #   Atomic units: Hartrees/charge or Hartrees/e

            if not flags_dict.get('esp'):
                psi4.set_output_file(f'{file_basename}-psi.out')

                psi4.core.set_active_molecule(conf)

                print('PSI4: ', f'"{flags_dict['basis_esp']}"')
                # psi4.set_options({'basis': f'"{flags_dict['basis_esp']}"'})  # fine for 1 basis set
                psi4.basis_helper('\n'.join(flags_dict['basis_esp']))  # good for 1 or a mix of basis sets
                psi4.set_options(flags_dict.get('psi4_options', {}))

                conf.set_molecular_charge(flags_dict['formal_charge'])
                conf.set_multiplicity(flags_dict['multiplicity'])

                psi4.prop(flags_dict['method_esp'], properties=['grid_esp'])
                psi4.core.clean()

                os.system(f"mv grid.dat {file_basename}_grid.dat")
                os.system(f"mv grid_esp.dat {file_basename}_esp.dat")

                data['esp_values'].append(np.loadtxt(f"{file_basename}_esp.dat"))
            else:
                data['esp_values'].append(np.loadtxt(flags_dict['esp'][conf_n]))

            # Build a matrix of the inverse distance from each ESP point to each nucleus.
            print(f"Points: {len(points)}; Coord: {len(data['coordinates'][conf_n])}")

            inverse_dist = np.zeros((len(points), len(data['coordinates'][conf_n])))

            for i in range(inverse_dist.shape[0]):
                for j in range(inverse_dist.shape[1]):
                    inverse_dist[i, j] = 1 / np.linalg.norm(points[i] - data['coordinates'][conf_n][j])

            data['inverse_dist'].append(inverse_dist * bohr_to_angstrom)  # convert to atomic units
            data['coordinates'][conf_n] /= bohr_to_angstrom  # convert to angstroms

            data['formal_charge'] = flags_dict['formal_charge']

        # Fit partial atomic charges.
        data = espfit.fit(options=flags_dict, data=data)

        write_results(flags_dict=flags_dict, data=data, output_file=output_file)

        return data['fitted_charges']
