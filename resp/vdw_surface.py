"""
A srcipt to generate van der Waals surface of molecules.
"""

from __future__ import division, absolute_import, print_function

import numpy as np


def vdw_radii(element: str) -> float:
    """ Assign the van der Waals radius to an element.
          The values were taken from GAMESS.
        Args
            element : one or two letter element identifier
        Returns
            van der Waals radius (Angstrom)
    """

    if not isinstance(element, str):
        raise TypeError(f'The element was not given as a string (i.e., {element} variable).')
    else:
        radii = {'H':  1.20, 'HE': 1.20,
                 'LI': 1.37, 'BE': 1.45, 'B':  1.45, 'C':  1.50,
                 'N':  1.50, 'O':  1.40, 'F':  1.35, 'NE': 1.30,
                 'NA': 1.57, 'MG': 1.36, 'AL': 1.24, 'SI': 1.17,
                 'P':  1.80, 'S':  1.75, 'CL': 1.70}

        if element in radii.keys():
            return radii[element]
        else:
            raise KeyError(f'{element} is not internally supported; use the "vdw_radii" option to add its vdw radius.')


def surface(n_points: int) -> np.ndarray:
    """ Computes approximately n points on a unit sphere.

        Code was adapted from GAMESS.

        Args
            n_points : number of surface points

        Returns
            xyz coordinates of surface points

        Library dependencies
            numpy
    """

    if not isinstance(n_points, int):
        raise TypeError(f'The number of points was not given as an integer (i.e., {n_points} variable).')
    else:
        surface_points = []
        eps = 1e-10

        nequat = int(np.sqrt(np.pi * n_points))
        nvert = int(nequat / 2)
        nu = 0

        for i in range(nvert + 1):
            fi = np.pi * i / nvert
            z = np.cos(fi)
            xy = np.sin(fi)
            nhor = int(nequat * xy + eps)

            if nhor < 1:
                nhor = 1

            for j in range(nhor):
                fj = 2 * np.pi * j / nhor
                x = np.cos(fj) * xy
                y = np.sin(fj) * xy

                if nu >= n_points:
                    return np.array(surface_points)

                nu += 1
                surface_points.append([x, y, z])

        return np.array(surface_points)


def vdw_surface(coordinates: np.ndarray, element_list: list, scale_factor: float,
                density: float, radii: dict) -> np.ndarray:
    """ Computes a molecular surface at points extended from the atoms' van der
        Waals radii.

        This is done using the Connolly [1] approach. As stated by Besler et al. [2],
        "Such a method is the surface generation algorithm of Connolly. It
        computes a spherical surface of points around each atom at a specified
        multiple of the atoms' van der Waals radius and density. The molecular
        surface is then constructed by taking the union of all of the atom
        surfaces and eliminating those points that are within the specified
        multiple of the van derWaals radius of any of the atoms."

        Args
            coordinates  : cartesian coordinates of the nuclei (Angstroms)
            element_list : element symbols (e.g., C, H)
            scale_factor : scaling factor - the points on the molecular surface are
                           set at a distance of scale_factor * vdw_radius away from
                           each of the atoms. Recommended scaling factors are
                           1.4, 1.6, 1.8, 2.0 [3]
            density      : The (approximate) number of points to generate per Angstrom^2
                           of surface area. 1.0 is recommended [2].
            radii        : VDW radii

        Returns
            surface_points (ndarray) : coordinates of the points on the extended surface

        Library dependencies
            numpy

        References:
        1. Connolly, M. L. Analytical Molecular-surface Calculation Journal of Applied
            Crystallography, 1983, 16, 548-558
        2. Besler, B. H.; Merz Jr., K. M. & Kollman, P. A. Atomic charges derived from
            semiempirical methods J. Comput. Chem., 1990, 11, 431-439
        3. Singh, U. C. & Kollman, P. A. An approach to computing electrostatic charges
            or molecules J. Comput. Chem., John Wiley & Sons, Ltd, 1984, 5, 129-145

        Also related:
        4. Bayly, C. I.; Cieplak, P.; Cornell, W. & Kollman, P. A. A well-behaved
            electrostatic potential based method using charge restraints for deriving
            atomic charges: the RESP model J. Phys. Chem., 1993, 97, 10269-10280
    """

    if not isinstance(coordinates, np.ndarray):
        raise TypeError(f'The coordinate were not given as an ndarray (i.e., {coordinates} variable).')
    elif not isinstance(element_list, list):
        raise TypeError(f'The elements were not given as a list (i.e., {element_list} variable).')
    elif not isinstance(scale_factor, float):
        raise TypeError(f'The scale_factor was not given as a float (i.e., {scale_factor} variable).')
    elif not isinstance(density, float):
        raise TypeError(f'The density was not given as a float (i.e., {density} variable).')
    elif not isinstance(radii, dict):
        raise TypeError(f'The radii were not given as a dictionary (i.e., {radii} variable).')
    else:
        radii_scaled = {}
        surface_points = []

        # scale radii
        for element in element_list:
            radii_scaled[element] = radii[element] * scale_factor

        # loop over atomic coordinates
        # for element, coordinate in zip(element_list, coordinates): # TODO, alternative approach - test later.
        for i in range(len(coordinates)):

            # calculate approximate number of ESP grid points
            n_points = int(density * 4.0 * np.pi * np.power(radii_scaled[element_list[i]], 2))  # why 4.0?

            # generate an array of n_points in a unit sphere around the atom
            dots = surface(n_points=n_points)

            # scale the unit sphere by the VDW radius and translate
            dots = coordinates[i] + radii_scaled[element_list[i]] * dots

            # determine which points should be included or removed due to overlaps
            for j in range(len(dots)):
                save = True
                for k in range(len(coordinates)):
                    if i == k:
                        continue

                    # exclude points within the scaled VDW radius of other atoms
                    d = np.linalg.norm(dots[j] - coordinates[k])

                    if d < radii_scaled[element_list[k]]:
                        save = False
                        break
                if save:
                    surface_points.append(dots[j])

        # could also return radii_scaled if desired
        return np.array(surface_points)
