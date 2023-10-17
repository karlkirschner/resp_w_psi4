"""
Driver for the RESP code.
"""
from __future__ import division, absolute_import, print_function

# Original Work:
__authors__   =  "Asem Alenaizan"
__credits__   =  ["Asem Alenaizan"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2018-04-28"

# Modified Work:
__authors__   =  ["Asem Alenaizan", "Karl N. Kirschner"]
__credits__   =  ["Asem Alenaizan", "Karl N. Kirschner"]
__date__      = "2023"


import configparser
import os

import numpy as np
import psi4
import espfit
import vdw_surface


bohr_to_angstrom = 0.52917721092


def write_results(flags_dict: dict, radii: list, scale_factor, molecules, data, notes, fitting_methods, qf):
    ''' Write out the results to disk
    '''

    with open("results.out", "w") as outfile:
        outfile.write("Electrostatic potential parameters\n")
        if flags_dict['grid'] == 'None':
            outfile.write("    van der Waals radii (Angstrom):\n")
            for i, j in radii.items():
                outfile.write(f"{' ':38s}{i} = {j/scale_factor:.3f}\n")
            outfile.write(f"    VDW scale factors: {' ':14s} ")
            for i in flags_dict["vdw_scale_factors"]:
                outfile.write(f"{i} ")
            outfile.write(f"\n    VDW point density: {' ':14s} {flags_dict['vdw_point_density']:.3f}\n")
        if flags_dict['esp'] == 'None':
            outfile.write(f"    ESP method: {' ':21s} {flags_dict['method_esp']}\n")
            outfile.write(f"    ESP basis set: {' ':18s} {flags_dict['basis_esp']}\n")

        for conf_n in range(len(flags_dict['input_files'])):
            with open(flags_dict['input_files'][conf_n]) as f:
                conf_xyz = f.read()

            molec_name = os.path.splitext(flags_dict['input_files'][conf_n])[0]
            conf = psi4.core.Molecule.from_string(conf_xyz, dtype='xyz', name=molec_name)

            outfile.write("\nGrid information (see %i_%s_grid.dat in %s)\n"
                    %(conf_n+1, conf.name(), conf.units()))
            outfile.write(f"    Number of grid points: {' ':10s} {len(data['esp_values'][conf_n])}\n")
            outfile.write(f"\nQuantum electrostatic potential (see {conf_n+1}_{conf.name()}_grid_esp.dat)\n")

        outfile.write("\nConstraints\n")
        if flags_dict['constraint_charge'] != 'None':
            outfile.write("    Charge constraints\n")
            outfile.write(f"        Total charge of {flags_dict['constraint_charge']}")  # %12.8f on the set" %i[0])
                # for j in i[1]:
                #     outfile.write("%4d" %j)
            outfile.write("\n")

        if flags_dict['constraint_group'] != 'None':
            outfile.write("    Equality constraints\n")
            outfile.write("        Equal charges on atoms\n")
            for i in flags_dict['constraint_group']:
                outfile.write("                               ")
                for j in i:
                    outfile.write("%4d" %j)
                outfile.write("\n")

        outfile.write("\nRestraint\n")
        if flags_dict['restraint']:
            outfile.write("    Hyperbolic restraint to a charge of zero\n")
            if flags_dict['ihfree']:
                outfile.write("    Hydrogen atoms are not restrained\n")
            outfile.write(f"    resp_a:  {' ':23s}  {flags_dict['resp_a']:.4f}\n")
            outfile.write(f"    resp_b:  {' ':24s} {flags_dict['resp_b']:.4f}\n")

        outfile.write("\nFit\n")
        outfile.write(f"{str(notes)}\n")

        outfile.write("\nElectrostatic Potential Charges\n")
        outfile.write("   Center  Symbol")
        for i in fitting_methods:
            outfile.write("%10s" %i)
        outfile.write("\n")
        for i in range(len(data['symbols'])):
            # outfile.write("  %5d    %s     " %(i+1, data['symbols'][i]))
            outfile.write(f"{i+1:6d}       {data['symbols'][i]}    ")
            for j in qf:
                outfile.write("%12.8f" %j[i])
            outfile.write("\n")

        outfile.write("\nTotal Charge:     ")
        for i in qf:
            outfile.write(f"{np.sum(i):12.8f}")
        outfile.write('\n')


def parse_ini(input_ini):
    ''' Process the input configuration file.

        Input:
            input_config: a <class 'configparser.ConfigParser'> object

        Output:
            dictionary of needed flags
    '''

    config = configparser.ConfigParser()
    config.read(input_ini)

    # flags = ['vdw_scale_factors',
    #          'vdw_point_density',
    #          'esp',
    #          'grid',
    #          'weight',
    #          'restraint',
    #          'resp_a',
    #          'resp_b',
    #          'ihfree',
    #          'toler',
    #          'max_it',
    #          'method_esp',
    #          'basis_esp'
    #         ]

    # flags_dict = dict.fromkeys(flags)
    flags_dict = {}

    # assign values to all keys
    for section in config:
        for key in config[section]:
            flags_dict[key] = config.get(section, key)

            if key == 'input_files':  # good for any list of strings
                flags_dict[key] = [item.strip("'") for item in flags_dict[key].split(",")]

            for key_float in ['esp', 'grid', 'constraint_charge']:
                if (key == key_float) and (flags_dict[key] != 'None'):
                    flags_dict[key] = [float(item.strip("'")) for item in flags_dict[key].split(",")]

            for key_float in ['vdw_scale_factors', 'weight']: # any list of float
                if key == key_float:  # good for any list of floats
                    flags_dict[key] = [float(item.strip("'")) for item in flags_dict[key].split(",")]

            for key_float in ['vdw_radii']:
                if key == key_float:
                    flags_dict[key] = {}

            for key_float in ['vdw_point_density', 'resp_a', 'resp_b', 'toler']: # any single float
                if key == key_float:  # good for any list of floats
                    flags_dict[key] = float(flags_dict[key])

            for key_float in ['max_it']: # any single int
                if key == key_float:  # good for any list of floats
                    flags_dict[key] = int(flags_dict[key])

            for key_float in ['restraint', 'ihfree']: # any single bool
                if key == key_float:  # good for any list of floats
                    flags_dict[key] = bool(flags_dict[key])

    return flags_dict


# def resp(molecules, options=None):
def resp(input_ini) -> list:
    """ RESP code driver.

    Args
        input_ini : input configuration for the calculation

    Returns
        charges : charges

    Notes
        output files : mol_results.dat: fitting results
                       mol_grid.dat: grid points in molecule.units
                       mol_grid_esp.dat: QM esp valuese in a.u. 
    """

    if not isinstance(input_ini, str):
        raise TypeError(f'The input configparser file (i.e., {input_ini}) is not a str.')


    flags_dict = parse_ini(input_ini)

    print('KNK flags:', flags_dict)

    # if options is None:
    #     options = {}

    # # Check options
    # # RESP options have large case keys
    # options = {k.upper(): v for k, v in sorted(options.items())}

    # # VDW surface options
    # if 'ESP' not in options:
    #     options['ESP'] = []
    # if 'GRID' not in options:
    #     options['GRID'] = []
    # if 'VDW_SCALE_FACTORS' not in options:
    #     options['VDW_SCALE_FACTORS'] = [1.4, 1.6, 1.8, 2.0]
    # if 'VDW_POINT_DENSITY' not in options:
    #     options['VDW_POINT_DENSITY'] = 1.0
    
    # # Hyperbolic restraint options
    # if 'WEIGHT' not in options:
    #     options['WEIGHT'] = [1]*len(molecules)
    # if 'RESTRAINT' not in options:
    #     options['RESTRAINT'] = True
    # if options['RESTRAINT']:
    #     if 'RESP_A' not in options:
    #         options['RESP_A'] = 0.0005
    #     if 'RESP_B' not in options:
    #         options['RESP_B'] = 0.1
    #     if 'IHFREE' not in options:
    #         options['IHFREE'] = True
    #     if 'TOLER' not in options:
    #         options['TOLER'] = 1e-5
    #     if 'MAX_IT' not in options:
    #         options['MAX_IT'] = 25

    # # QM options
    # if 'METHOD_ESP' not in options:
    #     options['METHOD_ESP'] = 'scf'
    # if 'BASIS_ESP' not in options:
    #     options['BASIS_ESP'] = '6-31g*'

    # # VDW surface options
    # if 'VDW_RADII' not in options:
    #     options['VDW_RADII'] = {}
    # radii = {}
    # for i in options['VDW_RADII']:
    #     radii[i.upper()] = options['VDW_RADII'][i]
    # options['VDW_RADII'] = radii

    # # Constraint options
    # if 'CONSTRAINT_CHARGE' not in options:
    #     options['CONSTRAINT_CHARGE'] = []
    # if 'CONSTRAINT_GROUP' not in options:
    #     options['CONSTRAINT_GROUP'] = []

    # Same data for all conformer
    data = {}

    with open(flags_dict['input_files'][0]) as infile:
        conformer_1_xyz = infile.read()

    molec_name = os.path.splitext(flags_dict['input_files'][0])[0]
    conf_1 = psi4.core.Molecule.from_string(conformer_1_xyz, dtype='xyz', name=molec_name)

    data['natoms'] = conf_1.natom()

    data['symbols'] = []
    for i in range(data['natoms']):
        data['symbols'].append(conf_1.symbol(i))

    data['mol_charge'] = conf_1.molecular_charge()

    # Data for each conformer
    data['coordinates'] = []
    data['esp_values'] = []
    data['invr'] = []

    for conf_n in range(len(flags_dict['input_files'])):

        with open(flags_dict['input_files'][conf_n]) as f:
            conf_xyz = f.read()

        molec_name = os.path.splitext(flags_dict['input_files'][conf_n])[0]

        conf = psi4.core.Molecule.from_string(conf_xyz, dtype='xyz', name=molec_name)
        coordinates = conf.geometry()
        coordinates = coordinates.np.astype('float')*bohr_to_angstrom
        data['coordinates'].append(coordinates)

        if flags_dict['grid'] != 'None':
            # Read grid points
            points = np.loadtxt(flags_dict['grid'][conf_n])
            np.savetxt('test.dat', points, fmt='%15.10f')
            if 'Bohr' in str(conf.units()):
                points *= bohr_to_angstrom

        else:
            # Get the points at which we're going to calculate the ESP
            points = []
            for scale_factor in flags_dict['vdw_scale_factors']:
                shell, radii = vdw_surface.vdw_surface(coordinates,
                                                       data['symbols'],
                                                       scale_factor,
                                                       flags_dict['vdw_point_density'],
                                                       flags_dict['vdw_radii'])
                points.append(shell)

            points = np.concatenate(points)

            if 'Bohr' in str(conf.units()):
                points /= bohr_to_angstrom
                np.savetxt('grid.dat', points, fmt='%15.10f')
                points *= bohr_to_angstrom
            else:
                np.savetxt('grid.dat', points, fmt='%15.10f')

        # Calculate ESP values at the grid
        if flags_dict['esp'] != 'None':
            data['esp_values'].append(np.loadtxt(flags_dict['esp'][conf_n]))
            np.savetxt('grid_esp.dat', data['esp_values'][-1], fmt='%15.10f')
        else:
            psi4.core.set_active_molecule(conf)
            psi4.set_options({'basis': flags_dict['basis_esp']})
            psi4.set_options(flags_dict.get('psi4_options', {}))   ### TODO: Figure this out
            psi4.prop(flags_dict['method_esp'], properties=['grid_esp'])
            data['esp_values'].append(np.loadtxt('grid_esp.dat'))
            psi4.core.clean()

        os.system(f"mv grid.dat {conf_n+1}_{conf.name()}_grid.dat")
        os.system(f"mv grid_esp.dat {conf_n+1}_{conf.name()}_grid_esp.dat")

        # Build a matrix of the inverse distance from each ESP point to each nucleus
        invr = np.zeros((len(points), len(coordinates)))
        for i in range(invr.shape[0]):
            for j in range(invr.shape[1]):
                invr[i, j] = 1/np.linalg.norm(points[i]-coordinates[j])

        data['invr'].append(invr*bohr_to_angstrom)  # convert to atomic units
        data['coordinates'][-1] /= bohr_to_angstrom # convert to angstroms

    # Calculate charges
    # qf, labelf, notes = espfit.fit(options=flags_dict, data=data)
    q_fitted, fitting_methods, notes = espfit.fit(options=flags_dict, data=data)

    write_results(flags_dict=flags_dict, radii=radii, scale_factor=scale_factor, molecules=conf, data=data, notes=notes, fitting_methods=fitting_methods, qf=q_fitted)

    return q_fitted
