"""
Driver for the RESP code.
"""
from __future__ import division, absolute_import, print_function

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
        with open(output_file, "w") as outfile:
            outfile.write("Electrostatic potential parameters\n")

            outfile.write("    van der Waals radii (Angstrom):\n")
            for element, radius in data['vdw_radii'].items():
                outfile.write(f"{' ':38s}{element} = {radius:.3f}\n")

            outfile.write(f"    VDW scale factors: {' ':14s} ")
            for i in flags_dict["vdw_scale_factors"]:
                outfile.write(f"{i} ")

            outfile.write(f"\n    VDW point density: {' ':14s} {flags_dict['vdw_point_density']:.3f}\n")

            if flags_dict['esp'] == 'None':
                outfile.write(f"    ESP method: {' ':21s} {flags_dict['method_esp']}\n")
                outfile.write(f"    ESP basis set: {' ':18s} {flags_dict['basis_esp']}\n")

            outfile.write(f'\nGrid information\n')
            outfile.write(f'    Quantum ESP File(s)\n')
            for conf_n in range(len(flags_dict['input_files'])):
                outfile.write(f"{' ':38s}{data['name'][conf_n]}_grid_esp.dat\n")

            outfile.write(f'    Grid Points File(s) (# of points)\n')
            for conf_n in range(len(flags_dict['input_files'])):
                outfile.write(f"{' ':38s}{data['name'][conf_n]}_grid.dat ({len(data['esp_values'][conf_n])})\n")

            outfile.write('\nConstraints\n')
            if flags_dict['constraint_charge'] != 'None':
                outfile.write('    Charge constraints\n')
                for key, value in flags_dict['constraint_charge'].items():
                    outfile.write(f"{' ':37s} Atom {key} = {value}\n")
            else:
                outfile.write(f"{' ':38s}None\n")

            if flags_dict['equivalent_groups'] != 'None':
                outfile.write('    Equivalent charges on atoms\n')
                count = 1
                for i in flags_dict['equivalent_groups']:
                    outfile.write(f"{' ':37s} group_{count} = ")
                    for j in i:
                        outfile.write(f'{j} ')
                    count += 1
                    outfile.write('\n')

            outfile.write('\nRestraint\n')
            if flags_dict['restraint']:
                outfile.write('    Hyperbolic restraint to a charge of zero\n')
                if flags_dict['ihfree']:
                    outfile.write('    Hydrogen atoms are not restrained\n')
                outfile.write(f"    resp_a:  {' ':23s}  {flags_dict['resp_a']:.4f}\n")
                outfile.write(f"    resp_b:  {' ':24s} {flags_dict['resp_b']:.4f}\n")

            outfile.write('\nFit\n')
            print('KNK', data['warnings'])
            if len(data['warnings']) > 0:
                outfile.write('   WARNINGS\n')
                for i in data['warnings']:
                    outfile.write(f"{' ':8s}{i}\n")

            outfile.write(f"\n{' ':4s}Electrostatic Potential Charges\n")
            outfile.write(f"{' ':8s}Center  Symbol{' ':7s}")
            if len(data["fitting_methods"]) > 0:
                for i in data["fitting_methods"]:
                    outfile.write(f"{i:12s}")
                outfile.write("\n")

            for i in range(len(data['symbols'])):
                outfile.write(f"{' ':10s}{i + 1}{' ':8s}{data['symbols'][i]}")
                outfile.write(f"{' ':4s}{data['fitted_charges'][0][i]:12.8f}{data['fitted_charges'][1][i]:12.8f}")
                outfile.write("\n")

            outfile.write(f"\n{' ':8s}Total Charge:{' ':3s}")
            for i in data['fitted_charges']:
                outfile.write(f"{np.sum(i):12.8f}")
            outfile.write('\n')


def parse_ini(input_ini: str) -> dict:
    """ Process the input configuration file.

        Args
            input_ini: a string that corresponds to a configparser ini file.

        Return
            flags_dict: all flags processed from input_ini

        Library dependencies
            configparser
    """

    if not isinstance(input_ini, str):
        raise TypeError(f'The input flags were not given as a dictionary (i.e., {input_ini}).')
    else:
        config = configparser.ConfigParser()
        config.read(input_ini)

        # flags_dict = dict.fromkeys(flags)
        flags_dict = {}

        # assign values to all keys
        for section in config:
            for key in config[section]:
                flags_dict[key] = config.get(section, key)

                if key == 'input_files':
                    flags_dict[key] = [item.strip("'").replace('\n', '').replace(' ', '')
                                       for item in flags_dict[key].split(',')]

                elif key == 'constraint_charge' and (flags_dict[key] != 'None'):
                    constraint_q_list = (constraints.replace('\n', '').replace(' ', '').split('=')
                                         for constraints in flags_dict[key].split(','))
                    flags_dict[key] = {int(atom_number): float(value) for atom_number, value in constraint_q_list}

                elif key == 'equivalent_groups' and (flags_dict[key] != 'None'):
                    all_groups = []
                    for constraints in flags_dict[key].replace('\n', '').split(','):
                        group = constraints.split('=')
                        atom_list = []
                        atom_list.extend(atom_number for atom_number in group[1].split())
                        atom_list = [int(x) for x in list(filter(None, atom_list))]  # remove empty strings, ensure int
                        all_groups.append(atom_list)
                    flags_dict[key] = all_groups

                for item in ['esp', 'grid']:
                    if (key == item) and (flags_dict[key] != 'None'):
                        flags_dict[key] = [str(item.strip("'")) for item in flags_dict[key].replace(' ', '').split(',')]

                for item in ['vdw_scale_factors', 'weight']:
                    if (key == item) and (flags_dict[key] != 'None'):
                        flags_dict[key] = [float(item.strip("'")) for item in flags_dict[key].replace(' ', '').split(',')]

                for item in ['vdw_radii']:
                    if (key == item) and (flags_dict[key] != 'None'):
                        radii_list = (atom_radius.replace('\n', '').replace(' ', '').split('=')
                                      for atom_radius in flags_dict[key].split(','))
                        flags_dict[key] = {element: float(radius) for element, radius in radii_list}

                for item in ['vdw_point_density', 'resp_a', 'resp_b', 'toler']:
                    if (key == item) and (flags_dict[key] != 'None'):
                        flags_dict[key] = float(flags_dict[key])

                for item in ['max_it']:
                    if (key == item) and (flags_dict[key] != 'None'):
                        flags_dict[key] = int(flags_dict[key])

                for item in ['restraint', 'ihfree']:
                    if (key == item) and (flags_dict[key] != 'None'):
                        if flags_dict[key] == 'False':
                            flags_dict[key] = False
                        else:
                            flags_dict[key] = True

        return flags_dict


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
        with open(infile) as input_file:
            molecule_xyz = input_file.read()

        molec_name = os.path.splitext(infile)[0]
        data_dict['name'].append(molec_name)

        molecule = psi4.core.Molecule.from_string(molecule_xyz, dtype='xyz', name=molec_name)

        data_dict['natoms'] = molecule.natom()

        data_dict['symbols'] = []
        for i in range(data_dict['natoms']):
            data_dict['symbols'].append(molecule.symbol(i))

        data_dict['mol_charge'] = molecule.molecular_charge()

        coordinates = molecule.geometry()
        coordinates = coordinates.np.astype('float') * bohr_to_angstrom
        data_dict['coordinates'].append(coordinates)

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
        output_file = input_ini.replace('ini', 'out')

        print('\nYour flags:', flags_dict)

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

            print("UNITS", conf.units())
            points = []  # units: Bohr
            if flags_dict['grid'] != 'None':
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
            if flags_dict['esp'] == 'None':
                psi4.core.set_active_molecule(conf)
                psi4.set_options({'basis': flags_dict['basis_esp']})
                psi4.set_options(flags_dict.get('psi4_options', {}))  # TODO: investigate more
                psi4.prop(flags_dict['method_esp'], properties=['grid_esp'])
                psi4.core.clean()

                os.system(f"mv grid.dat {file_basename}_grid.dat")
                os.system(f"mv grid_esp.dat {file_basename}_esp.dat")

            data['esp_values'].append(np.loadtxt(f"{file_basename}_esp.dat"))

            # Build a matrix of the inverse distance from each ESP point to each nucleus
            inverse_dist = np.zeros((len(points), len(data['coordinates'][conf_n])))
            for i in range(inverse_dist.shape[0]):
                for j in range(inverse_dist.shape[1]):
                    inverse_dist[i, j] = 1 / np.linalg.norm(points[i] - data['coordinates'][conf_n][j])

            data['inverse_dist'].append(inverse_dist * bohr_to_angstrom)  # convert to atomic units
            data['coordinates'][conf_n] /= bohr_to_angstrom  # convert to angstroms

        print('KNK', len(data['esp_values']), len(points))
        print('POINTS', points)
        # Calculate charges
        data = espfit.fit(options=flags_dict, data=data)

        write_results(flags_dict=flags_dict, data=data, output_file=output_file)

        return data['fitted_charges']
