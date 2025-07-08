"""
Driver for the RESP code.
"""

# Original Work:
__authors__ = "Asem Alenaizan"
__credits__ = ["Asem Alenaizan"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-04-28"

# Modified Work:
__authors__ = ["Karl N. Kirschner"]
__credits__ = ["Karl N. Kirschner"]
__date__ = "2023-11"

import ast           # For safely evaluating list/dict literals from strings
import configparser
import os

import numpy as np
import psi4

import espfit
import vdw_surface

bohr_to_angstrom = 0.52917721092


def write_results(flags_dict: dict, data: dict, output_file: str):
    """ Write out the results to disk.

        Args
            flags_dict  : input flags for calculation
            data        : data that has been computed for molecule
            output_file : name of output file
        Return
            Output text file
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
            if 'esp' in data['fitting_methods'] and 'esp_rmse_true' in data:
                outfile.write(f"{data['esp_rmse_true']:12.5f}")
            else:
                outfile.write(f"{'N/A':>12s}") # If ESP RMSE not available
            
            if 'resp' in data['fitting_methods'] and 'resp_rmse_true' in data:
                outfile.write(f"{data['resp_rmse_true']:12.5f}")
            else:
                outfile.write(f"{'N/A':>12s}") # If RESP RMSE not available
            outfile.write('\n')

            # RRMS row
            outfile.write(f"{' ':8s}{'RRMS':<10s}")
            if 'esp' in data['fitting_methods'] and 'esp_rrms_true' in data:
                outfile.write(f"{data['esp_rrms_true']:12.5f}")
            else:
                outfile.write(f"{'N/A':>12s}") # If ESP RRMS not available

            if 'resp' in data['fitting_methods'] and 'resp_rrms_true' in data:
                outfile.write(f"{data['resp_rrms_true']:12.5f}")
            else:
                outfile.write(f"{'N/A':>12s}") # If RESP RRMS not available
            outfile.write('\n')


##KNK's original version
# def write_results(flags_dict: dict, data: dict, output_file: str):
#     """ Write out the results to disk.

#         Args
#             flags_dict  : input flags for calculation
#             data        : data that has been computed for molecule
#             output_file : name of output file
#         Return
#             Output text file
#     """

#     if not isinstance(flags_dict, dict):
#         raise TypeError(f'The input flags were not given as a dictionary (i.e., {flags_dict}).')
#     elif not isinstance(data, dict):
#         raise TypeError(f'The resulting data were not given as a dictionary (i.e., {data}).')
#     else:
#         with open(output_file, 'w') as outfile:
#             outfile.write("Electrostatic potential parameters\n")

#             outfile.write(f"{' ':4s}van der Waals radii (Angstrom):\n")
#             for element, radius in data['vdw_radii'].items():
#                 outfile.write(f"{' ':38s}{element} = {radius:.3f}\n")

#             vdw_scale_factors_str = ""
#             if flags_dict['vdw_scale_factors'] is not None:
#                 vdw_scale_factors_str = ' '.join([str(i) for i in flags_dict['vdw_scale_factors']])
#             else:
#                 vdw_scale_factors_str = "None" # Or "Not Set" or "Default"

#             outfile.write(f"{' ':4s}VDW scale factors:{' ':16s}{vdw_scale_factors_str}\n")

#             outfile.write(f"{' ':4s}VDW point density:{' ':16s}{flags_dict['vdw_point_density']}\n")

#             outfile.write(f"{' ':4s}ESP method:{' ':23s}{flags_dict['method_esp']}\n")

#             if flags_dict['basis_esp'] is None:
#                 outfile.write(f"{' ':4s}ESP basis set:{' ':20s}{flags_dict['basis_esp']}\n")
#             else:
#                 outfile.write(f"{' ':4s}ESP basis set:\n")
#                 for basis in flags_dict['basis_esp']:
#                     outfile.write(f"{' ':38s}{basis}\n")

#             outfile.write(f'\nGrid information\n')
#             outfile.write(f"{' ':4s}Quantum ESP File(s):\n")
#             if flags_dict['esp'] is None:
#                 for conf_n in range(len(flags_dict['input_files'])):
#                     outfile.write(f"{' ':38s}{data['name'][conf_n]}_grid_esp.dat\n")
#             else:
#                 for conf_n in range(len(flags_dict['input_files'])):
#                     outfile.write(f"{' ':38s}{flags_dict['esp'][conf_n]}\n")

#             outfile.write(f"\n{' ':4s}Grid Points File(s) (# of points):\n")
#             if flags_dict['grid'] is None:
#                 for conf_n in range(len(flags_dict['input_files'])):
#                     outfile.write(f"{' ':38s}{data['name'][conf_n]}_grid.dat ({len(data['esp_values'][conf_n])})\n")
#             else:
#                 for conf_n in range(len(flags_dict['input_files'])):
#                     outfile.write(f"{' ':38s}{flags_dict['grid'][conf_n]} ({len(data['esp_values'][conf_n])})\n")

#             outfile.write('\nConstraints\n')
#             if flags_dict['constraint_charge'] is not None:
#                 outfile.write(f"{' ':4s}Charge constraints:\n")
#                 for key, value in flags_dict['constraint_charge'].items():
#                     outfile.write(f"{' ':38s}Atom {key} = {value}\n")
#             else:
#                 outfile.write(f"{' ':4s}Charge constraints: {flags_dict['constraint_charge']}\n")

#             if flags_dict['equivalent_groups'] is not None:
#                 outfile.write(f"\n{' ':4s}Equivalent charges on atoms (group = atom numbers):\n")
#                 count = 1
#                 for i in flags_dict['equivalent_groups']:
#                     outfile.write(f"{' ':38s}group_{count} = ")
#                     for j in i:
#                         outfile.write(f'{j} ')
#                     count += 1
#                     outfile.write('\n')
#             else:
#                 outfile.write(f"\n{' ':4s}Equivalent charges on atoms: {flags_dict['equivalent_groups']}\n")

#             outfile.write('\nRestraint\n')
#             outfile.write(f"{' ':4s}ihfree:{' ':27s}{flags_dict['ihfree']:}\n")
#             outfile.write(f"{' ':4s}resp_a:{' ':27s}{flags_dict['resp_a']:.4f}\n")
#             outfile.write(f"{' ':4s}resp_b:{' ':27s}{flags_dict['resp_b']:.4f}\n")

#             outfile.write('\nFit\n')
#             if len(data['warnings']) > 0:
#                 outfile.write(f"{' ':4s}WARNINGS:\n")
#                 for i in data['warnings']:
#                     outfile.write(f"{' ':8s}{i}\n")
#                 outfile.write("\n")

#             outfile.write(f"{' ':4s}Electrostatic Potential Charges:\n")
#             outfile.write(f"{' ':8s}Center  Symbol{' ':8s}")
#             if len(data['fitting_methods']) > 0:
#                 for i in data['fitting_methods']:
#                     outfile.write(f'{i:12s}')
#                 outfile.write('\n')

#             for i in range(len(data['symbols'])):
#                 outfile.write(f"{' ':8s}{i + 1:3d}{' ':8s}{data['symbols'][i]:2s}")
#                 outfile.write(f"{' ':4s}{data['fitted_charges'][0][i]:12.8f}{data['fitted_charges'][1][i]:12.8f}\n")

#             outfile.write(f"\n{' ':8s}Total Charge:{' ':4s}")
#             for i in data['fitted_charges']:
#                 outfile.write(f'{np.sum(i):12.8f}')
#             outfile.write('\n')

#             # --- Fitting Statistics Section ---
#             outfile.write(f"\n{' ':4s}Fitting Statistics:\n")
#             outfile.write(f"{' ':8s}{'Metric':<10s}") # Left-align 'Metric'
            
#             # headers (ESP, RESP)
#             if 'esp' in data['fitting_methods']:
#                 outfile.write(f"{'ESP':>12s}")
#             if 'resp' in data['fitting_methods']:
#                 outfile.write(f"{'RESP':>12s}")
#             outfile.write('\n')

#             # RMSE row
#             outfile.write(f"{' ':8s}{'RMSE':<10s}")
#             if 'esp' in data['fitting_methods'] and 'esp_rmse_true' in data:
#                 outfile.write(f"{data['esp_rmse_true']:12.5f}")
#             else:
#                 outfile.write(f"{'N/A':>12s}")
            
#             if 'resp' in data['fitting_methods'] and 'resp_rmse_true' in data:
#                 outfile.write(f"{data['resp_rmse_true']:12.5f}")
#             else:
#                 outfile.write(f"{'N/A':>12s}")
#             outfile.write('\n')

#             # RRMS row
#             outfile.write(f"{' ':8s}{'RRMS':<10s}")
#             if 'esp' in data['fitting_methods'] and 'esp_rrms_true' in data:
#                 outfile.write(f"{data['esp_rrms_true']:12.5f}")
#             else:
#                 outfile.write(f"{'N/A':>12s}")

#             if 'resp' in data['fitting_methods'] and 'resp_rrms_true' in data:
#                 outfile.write(f"{data['resp_rrms_true']:12.5f}")
#             else:
#                 outfile.write(f"{'N/A':>12s}")
#             outfile.write('\n')


def parse_ini(input_ini: str) -> dict:
    """ Processes the input configuration file.

        Args:
            input_ini (str): A string that corresponds to a configparser ini file path.

        Returns:
            dict: All flags processed from input_ini, with appropriate type conversions.

        Library dependencies:
            configparser
            ast (for literal_eval)
    """
    if not isinstance(input_ini, str):
        raise TypeError(f'The input_ini must be a string (i.e., {input_ini}).')

    config = configparser.ConfigParser()
    config.read(input_ini)

    flags_dict = {}

    # Helper function to convert 'None' string to Python None
    def convert_none(value_str):
        return None if value_str.strip().lower() == 'none' else value_str

    # Iterate through all sections and keys
    for section in config.sections(): # Use config.sections() to avoid the DEFAULT section if it's not needed
        for key in config[section]:
            raw_value = config.get(section, key)
            processed_value = convert_none(raw_value) # Convert 'None' string to Python None first

            if processed_value is None:
                flags_dict[key] = None
                continue # Move to the next key if it's explicitly None

            # --- Type-specific parsing ---
            if key == 'input_files':
                # Split by comma, strip spaces, remove newlines (from multi-line values)
                flags_dict[key] = [item.strip().replace('\n', '') for item in processed_value.split(',')]


            elif key == 'constraint_charge':
                ## Original code
                # constraint_q_list = (constraints.replace('\n', '').replace(' ', '').split('=')
                #                      for constraints in processed_value.split(','))
                # flags_dict[key] = {int(atom_number): float(value) for atom_number, value in constraint_q_list}

                # Expects 'atom_number = partial_charge, atom_number = partial_charge, ...'

                parsed_constraints = {}
                # Split by comma to get individual constraints
                for constraint_str in processed_value.split(','):
                    constraint_str = constraint_str.strip()
                    if constraint_str: # Skip empty strings
                        try:
                            atom_number_str, value_str = constraint_str.split('=')
                            atom_number = int(atom_number_str.strip())
                            value = float(value_str.strip())
                            parsed_constraints[atom_number] = value
                        except ValueError as e:
                            print(f"Warning: Could not parse constraint_charge part '{constraint_str}'. Error: {e}")
                            # Optionally, you might want to raise an error or set a default
                            # For now, it will just skip malformed parts.
                flags_dict[key] = parsed_constraints

            elif key == 'equivalent_groups':
                all_groups = []
                for constraints in processed_value.replace('\n', '').split(','):
                    group = constraints.split('=')
                    atom_list = []
                    atom_list.extend(atom_number for atom_number in group[1].split())
                    atom_list = [int(x) for x in list(filter(None, atom_list))]  # remove empty strings, ensure int
                    all_groups.append(atom_list)
                flags_dict[key] = all_groups


            elif key in ['esp', 'grid']:
                # List of strings (file names)
                flags_dict[key] = [item.strip() for item in processed_value.replace(' ', '').replace('\n', '').split(',')]

            elif key in ['vdw_scale_factors', 'weight']:
                # List of floats
                flags_dict[key] = [float(item.strip()) for item in processed_value.replace(' ', '').replace('\n', '').split(',')]

            elif key == 'vdw_radii':
                # Dictionary of element: radius
                parsed_radii = {}
                for item_str in processed_value.split(','):
                    item_str = item_str.strip()
                    if item_str:
                        try:
                            element, radius = item_str.split('=')
                            parsed_radii[element.strip()] = float(radius.strip())
                        except ValueError as e:
                            print(f"Warning: Could not parse vdw_radii part '{item_str}'. Error: {e}")
                flags_dict[key] = parsed_radii

            elif key in ['vdw_point_density', 'resp_a', 'resp_b', 'toler']:
                # Float values
                try:
                    flags_dict[key] = float(processed_value)
                except ValueError as e:
                    print(f"Error parsing float for '{key}': '{processed_value}'. Error: {e}")
                    flags_dict[key] = None # Or raise error

            elif key in ['max_it', 'formal_charge', 'multiplicity']:
                # Integer values
                try:
                    flags_dict[key] = int(processed_value)
                except ValueError as e:
                    print(f"Error parsing int for '{key}': '{processed_value}'. Error: {e}")
                    flags_dict[key] = None # Or raise error

            elif key in ['restraint', 'ihfree']:
                # Boolean values
                flags_dict[key] = processed_value.lower() == 'true' # Converts 'True' to True, 'False' to False

            elif key == 'method_esp':
                # Simple string (already handled by default if not 'None')
                flags_dict[key] = processed_value

            elif key == 'basis_esp':
                for item in ['basis_esp']:
                    print(item, processed_value)
                    flags_dict[key] = [str(item.strip("'")) for item in processed_value.replace('\n', '').split(',')]

            else:
                print('TEST OTHER')
                # Default for any unhandled keys: keep as string
                flags_dict[key] = processed_value

    return flags_dict


## Original Code:
# def parse_ini(input_ini: str) -> dict:
#     """ Process the input configuration file.

#         Args
#             input_ini: a string that corresponds to a configparser ini file.

#         Return
#             flags_dict: all flags processed from input_ini

#         Library dependencies
#             configparser
#     """

#     if not isinstance(input_ini, str):
#         raise TypeError(f'The input flags were not given as a dictionary (i.e., {input_ini}).')
#     else:
#         config = configparser.ConfigParser()
#         config.read(input_ini)

#         # flags_dict = dict.fromkeys(flags)
#         flags_dict = {}

#         # assign values to all keys
#         for section in config:
#             for key in config[section]:
#                 flags_dict[key] = config.get(section, key)

#                 if key == 'input_files':
#                     flags_dict[key] = [item.strip("'").replace('\n', '').replace(' ', '')
#                                        for item in flags_dict[key].split(',')]

#                 elif key == 'constraint_charge' and (flags_dict[key] != 'None'):
#                     constraint_q_list = (constraints.replace('\n', '').replace(' ', '').split('=')
#                                          for constraints in flags_dict[key].split(','))
#                     flags_dict[key] = {int(atom_number): float(value) for atom_number, value in constraint_q_list}

#                 elif key == 'equivalent_groups' and (flags_dict[key] != 'None'):
#                     all_groups = []
#                     for constraints in flags_dict[key].replace('\n', '').split(','):
#                         group = constraints.split('=')
#                         atom_list = []
#                         atom_list.extend(atom_number for atom_number in group[1].split())
#                         atom_list = [int(x) for x in list(filter(None, atom_list))]  # remove empty strings, ensure int
#                         all_groups.append(atom_list)
#                     flags_dict[key] = all_groups

#                 for item in ['esp', 'grid']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         flags_dict[key] = [str(item.strip("'")) for item in flags_dict[key].replace(' ', '').replace('\n', '').split(',')]

#                 for item in ['vdw_scale_factors', 'weight']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         flags_dict[key] = [float(item.strip("'")) for item in flags_dict[key].replace(' ', '').replace('\n', '').split(',')]

#                 for item in ['vdw_radii']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         radii_list = (atom_radius.replace('\n', '').replace(' ', '').split('=')
#                                       for atom_radius in flags_dict[key].split(','))
#                         flags_dict[key] = {element: float(radius) for element, radius in radii_list}

#                 for item in ['vdw_point_density', 'resp_a', 'resp_b', 'toler']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         flags_dict[key] = float(flags_dict[key])
#                     # else:
#                     #     print(flags_dict[key])
#                     #     flags_dict[key] = str(flags_dict[key])

#                 for item in ['max_it']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         flags_dict[key] = int(flags_dict[key])

#                 for item in ['restraint', 'ihfree']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         if flags_dict[key] == 'False':
#                             flags_dict[key] = False
#                         else:
#                             flags_dict[key] = True

#                 for item in ['method_esp']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         flags_dict[key] = str(flags_dict[key])

#                 for item in ['basis_esp']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         flags_dict[key] = [str(item.strip("'")) for item in flags_dict[key].replace('\n', '').split(',')]

#                 for item in ['formal_charge']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         flags_dict[key] = int(flags_dict[key])

#                 for item in ['multiplicity']:
#                     if (key == item) and (flags_dict[key] != 'None'):
#                         flags_dict[key] = int(flags_dict[key])

#     return flags_dict


def basic_molec_data(infile: str, data_dict: dict) -> dict:
    """ Extract basic data from input XYZ-formatted file.
            Basic data includes the following
                molecule's name
                number of atoms
                element symbols
                molecular charge

        Also creates a Psi4 molecule that is needed for QM calculations.

        Args
            infile: name of XYZ-formatted file
            data_dict: a dictionary for storing computed data

        Return
            data_dict: keys -> name, natoms, symbols, mol_charge
            molecule: a psi4 molecule object

        Library dependencies
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
        # print(f"INPUT DATA\n{data_for_psi4}\n{data_dict['symbols']}, {data_for_psi4}")

        data_dict['natoms'] = len(data_dict['symbols'])

        coordinates = np.float64(coordinates)
        print(f'INPUT COORDS, ANGSTROMS\n {coordinates}')
        # coordinates = coordinates / bohr_to_angstrom
        # print(f'INPUT COORDS, BOHR\n {coordinates}')

        molecule = psi4.core.Molecule.from_string(data_for_psi4, name=molec_name)  # note: will returns coords in Bohr
        # coordinates = molecule.geometry()
        # print(f"PSI4 COORDS, BOHR\n {coordinates.np.astype('float')}")
        # coordinates = coordinates.np.astype('float') * bohr_to_angstrom
        # print(f"PSI4 COORDS, ANGSTROMS\n {coordinates}")

        # print(data_dict['formal_charge'])
        # molecule.set_molecular_charge(data_dict['formal_charge'])
        # molecule.set_multiplicity(data_dict['multiplicity'])
        # data_dict['mol_charge'] = molecule.molecular_charge()
        # data_dict['mol_charge'] = -1
        # print("KNKNK", data_dict['mol_charge'])

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

        if (flags_dict['basis_esp'] is not None) and (flags_dict['esp'] is not None):
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

            data, conf = basic_molec_data(infile=flags_dict['input_files'][conf_n], data_dict=data)

            vdw_radii = {}  # units: Angstrom
            for element in data['symbols']:
                if element in flags_dict['vdw_radii']:
                    vdw_radii[element] = flags_dict['vdw_radii'][element]
                else:
                    # use built-in vdw_radii
                    vdw_radii[element] = vdw_surface.vdw_radii(element=element)

            data['vdw_radii'] = vdw_radii

            points = []  # units: Bohr
            if flags_dict['grid'] is not None:
                points = np.loadtxt(flags_dict['grid'][conf_n])

                if 'Bohr' in str(conf.units()):
                    points *= bohr_to_angstrom
            else:
                # compute grid points and save to file
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

            # Calculate ESP values along the grid points
            # From Manual, v. 1.10a1.dev86": "The grid.dat file is completely free form; any number of spaces and/or newlines between entries is permitted.
            #                                 The units of the coordinates in grid.dat are the same as those used to specify the molecule’s geometry,
            #                                 and the output quantities are always in atomic units."
            #   Atomic units: Hartrees/charge or Hartrees/e
            if flags_dict['esp'] is None:
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

            # Build a matrix of the inverse distance from each ESP point to each nucleus
            print(f"Points: {len(points)}; Coord: {len(data['coordinates'][conf_n])}")
            inverse_dist = np.zeros((len(points), len(data['coordinates'][conf_n])))
            for i in range(inverse_dist.shape[0]):
                for j in range(inverse_dist.shape[1]):
                    inverse_dist[i, j] = 1 / np.linalg.norm(points[i] - data['coordinates'][conf_n][j])

            data['inverse_dist'].append(inverse_dist * bohr_to_angstrom)  # convert to atomic units
            data['coordinates'][conf_n] /= bohr_to_angstrom  # convert to angstroms

            data['formal_charge'] = flags_dict['formal_charge']

        # Calculate charges
        data = espfit.fit(options=flags_dict, data=data)
        write_results(flags_dict=flags_dict, data=data, output_file=output_file)

        return data['fitted_charges']
