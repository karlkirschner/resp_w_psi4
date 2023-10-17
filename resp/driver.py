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
def resp(input_ini):
    """RESP code driver.

    Parameters
    ---------- 
    molecules : list
        list of psi4.Molecule instances
    options_list : dict, optional
        a dictionary of user's defined options

    Returns
    -------
    charges : list
        list of charges

    Note
    ----
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

    # # Write the results to disk
    # with open("results.out", "w") as f:
    #     f.write("Electrostatic potential parameters\n")
    #     if not options['GRID']:
    #         f.write("    van der Waals radii (Angstrom):\n")
    #         for i, j in radii.items():
    #             f.write("                               %8s%8.3f\n" %(i, j/scale_factor))
    #         f.write("    VDW scale factors:              ")
    #         for i in options["VDW_SCALE_FACTORS"]:
    #             f.write('%6.2f' %i)
    #         f.write('\n')
    #         f.write("    VDW point density:                %.3f\n" %(options["VDW_POINT_DENSITY"]))
    #     if not options['ESP']:
    #         f.write("    ESP method:                       %s\n" %options['METHOD_ESP'])
    #         f.write("    ESP basis set:                    %s\n" %options['BASIS_ESP'])

    #     for conf_n in range(len(molecules)):
    #         f.write("\nGrid information (see %i_%s_grid.dat in %s)\n"
    #                 %(conf_n+1, molecules[conf_n].name(), molecules[conf_n].units()))
    #         f.write("    Number of grid points:            %d\n" %len(data['esp_values'][conf_n]))
    #         f.write("\nQuantum electrostatic potential (see %i_%s_grid_esp.dat)\n" %(conf_n+1, molecules[conf_n].name()))

    #     f.write("\nConstraints\n")
    #     if options['CONSTRAINT_CHARGE']:
    #         f.write("    Charge constraints\n")
    #         for i in options['CONSTRAINT_CHARGE']:
    #             f.write("        Total charge of %12.8f on the set" %i[0])
    #             for j in i[1]:
    #                 f.write("%4d" %j)
    #             f.write("\n")
    #     if options['CONSTRAINT_GROUP']:
    #         f.write("    Equality constraints\n")
    #         f.write("        Equal charges on atoms\n")
    #         for i in options['CONSTRAINT_GROUP']:
    #             f.write("                               ")
    #             for j in i:
    #                 f.write("%4d" %j)
    #             f.write("\n")

    #     f.write("\nRestraint\n")
    #     if options['RESTRAINT']:
    #         f.write("    Hyperbolic restraint to a charge of zero\n")
    #         if options['IHFREE']:
    #             f.write("    Hydrogen atoms are not restrained\n")
    #         f.write("    resp_a:                           %.4f\n" %(options["RESP_A"]))
    #         f.write("    resp_b:                           %.4f\n" %(options["RESP_B"]))

    #     f.write("\nFit\n")
    #     f.write(notes)
    #     f.write("\nElectrostatic Potential Charges\n")
    #     f.write("   Center  Symbol")
    #     for i in labelf:
    #         f.write("%10s" %i)
    #     f.write("\n")
    #     for i in range(data['natoms']):
    #         f.write("  %5d    %s     " %(i+1, data['symbols'][i]))
    #         for j in qf:
    #             f.write("%12.8f" %j[i])
    #         f.write("\n")
    #     f.write("Total Charge:    ")
    #     for i in qf:
    #         f.write("%12.8f" %np.sum(i))
    #     f.write('\n')

    # return qf
