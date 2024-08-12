import argparse
import logging
import os
import shutil
from pathlib import Path

import numpy as np
from arkane.common import get_principal_moments_of_inertia, symbol_by_number
from arkane.encorr.corr import (
    assign_frequency_scale_factor,
    get_atom_correction,
    get_bac,
)
from arkane.modelchem import CompositeLevelOfTheory, LevelOfTheory
from rmgpy import constants
from rmgpy.qm.qmdata import QMData
from rmgpy.qm.symmetry import PointGroupCalculator
from rmgpy.molecule.element import get_element
from rmgpy.statmech import (
    Conformer,
    HarmonicOscillator,
    IdealGasTranslation,
    LinearRotor,
    NonlinearRotor,
)


def get_symmetry(coords, atom_numbers, scr_dir=None):

    scr_dir = scr_dir or os.path.join(".", "scratch")
    os.makedirs(scr_dir, exist_ok=True)

    symmetry = optical_isomers = 1
    try:
        qmdata = QMData(
            groundStateDegeneracy=1,  # Only needed to check if valid QMData
            numberOfAtoms=len(atom_numbers),
            atomicNumbers=atom_numbers,
            atomCoords=(coords, "angstrom"),
            energy=(0.0, "kcal/mol"),  # Only needed to avoid error
        )
        settings = type(
            "", (), dict(symmetryPath="symmetry", scratchDirectory=scr_dir)
        )()
        pgc = PointGroupCalculator(
            settings, "0", qmdata
        )  # '0' is an unique id used for calculator
        pg = pgc.calculate()
        if pg is not None:
            symmetry = pg.symmetry_number
            optical_isomers = 2 if pg.chiral else 1
            logging.debug(
                f"Symmetry algorithm found {optical_isomers} optical isomers "
                f"and a symmetry number of {symmetry}"
            )
        else:
            logging.warning(
                "Symmetry algorithm errored when computing point group. "
                "Using symmetry number=1 and optical isomers = 1 for "
                "further calculations, which may not be true."
            )
        return symmetry, optical_isomers
    finally:
        shutil.rmtree(scr_dir)


def get_lot_and_freq_scale(
    energy_level: str,
    freq_level: str,
    energy_software: str,
    freq_software: str,
    freq_scale: float,
):
    # Get energy level and assign software
    energy_method, energy_basis = energy_level.split("/")
    energy_level = LevelOfTheory(
        method=energy_method, basis=energy_basis, software=energy_software
    )

    # Get freq level
    freq_method, freq_basis = freq_level.split("/")
    freq_level = LevelOfTheory(
        method=freq_method, basis=freq_basis, software=freq_software
    )

    # Assign level of theory and frequency scale factor
    if energy_level.to_model_chem() in ["cbsqb3"] or energy_level == freq_level:
        level_of_theory = energy_level
    else:
        level_of_theory = CompositeLevelOfTheory(freq=freq_level, energy=energy_level)
    if freq_scale is None:
        try:
            freq_scale = assign_frequency_scale_factor(level_of_theory)
        except:
            logging.warning("Setting freq scale to 1.0")
            freq_scale = 1.0
    logging.warning(f"freq scale is {freq_scale}")
    return level_of_theory, freq_scale


def get_rotational_mode(coords, number, external_symmetry=None):
    if not external_symmetry:
        external_symmetry, _ = get_symmetry(coords, number)

    # Rotational
    moments_of_inertia = get_principal_moments_of_inertia(
        coords=coords,
        numbers=number,
    )[0]
    if any([moment_of_inertia == 0.0 for moment_of_inertia in moments_of_inertia]):
        # this is a linear rotor
        moments_of_inertia = [
            moment_of_inertia
            for moment_of_inertia in moments_of_inertia
            if moment_of_inertia != 0.0
        ]
        if abs(moments_of_inertia[0] - moments_of_inertia[1]) > 0.01:
            raise Exceptions(
                f"Expected two identical moments of inertia for a linear rigis rotor, "
                f"but got {moments_of_inertia}"
            )
        return LinearRotor(
            inertia=(moments_of_inertia[0], "amu*angstrom^2"),
            symmetry=external_symmetry,
        )
    else:
        # this is a non-linear rotor
        return NonlinearRotor(
            inertia=(moments_of_inertia, "amu*angstrom^2"), symmetry=external_symmetry
        )


def get_element_counts(number):
    # Get atoms count
    atoms = {}
    for atom_num in number:
        try:
            symbol = symbol_by_number[atom_num]
        except KeyError:
            raise ElementError(
                "Could not recognize element number {0}.".format(atom_num)
            )
        atoms[symbol] = atoms.get(symbol, 0) + 1
    return atoms


def get_rmg_conformer(
    label,
    level_of_theory,
    e_electronic,
    frequencies,
    coords,
    numbers,
    mass,
    multiplicity,
    freq_scale,
    molecule=None,
    use_atom_corrections=True,
    use_bond_corrections=False,
    bac_type="p",
    scr_dir=None,
):

    external_symmetry, optical_isomers = get_symmetry(coords, numbers, scr_dir=scr_dir)

    modes = []
    # Translational
    translation = IdealGasTranslation(mass=mass)
    modes.append(translation)

    # Rotational
    rotation = get_rotational_mode(coords, numbers, external_symmetry=external_symmetry)
    modes.append(rotation)

    # Vibrational
    frequencies = np.array(frequencies)
    frequencies = frequencies[frequencies >= 0]
    vibration = HarmonicOscillator(frequencies=(frequencies * freq_scale, "cm^-1"))
    modes.append(vibration)

    # Atom energy corrections
    if use_atom_corrections:
        atoms = get_element_counts(numbers)
        atom_corrections = get_atom_correction(level_of_theory, atoms)
    else:
        atom_corrections = 0

    # Bond energy corrections
    if use_bond_corrections:
        # Get bonds count
        if molecule is None:
            raise ValueError(
                "Cannot get bond corrections, since argument ``molecule`` is not provided."
            )

        bonds = molecule.enumerate_bonds()
        bond_corrections = get_bac(
            level_of_theory,
            bonds,
            coords,
            numbers,
            bac_type=bac_type,
            multiplicity=multiplicity,
        )

    else:
        bond_corrections = 0

    e_electronic_with_corrections = e_electronic + atom_corrections + bond_corrections

    if len(numbers) > 1:
        zpe_scale_factor = freq_scale / 1.014
        scaled_zpe = (
            0.5
            * constants.h
            * constants.c
            * constants.Na
            * np.sum(frequencies)
            * 100
            * zpe_scale_factor
        )
    else:
        scaled_zpe = 0

    e0 = ((e_electronic_with_corrections + scaled_zpe) * 0.001, "kJ/mol")

    return Conformer(
        E0=e0,
        modes=modes,
        spin_multiplicity=multiplicity,
        optical_isomers=optical_isomers,
    )


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, help=".csv file path containing parsed results", required=True
    )
    parser.add_argument(
        "--freq_level", type=str, help="Frequency level of theory", required=True
    )
    parser.add_argument(
        "--freq_scale", type=float, help="Optional frequency scale factor to use", required=True
    )
    parser.add_argument("--freq_software", type=str, help="Frequency software", required=True)
    parser.add_argument("--energy_level", type=str, help="Energy level of theory", required=True)
    parser.add_argument("--energy_software", type=str, help="Energy software", required=True)
    parser.add_argument("--no_bac_for_thermo", action="store_true", help="Do not use bond corrections for thermo")
    parser.add_argument(
        "--n_jobs", type=int, help="Number of jobs to run in parallel", default=1
    )
    parser.add_argument(
        "--save_path", type=str, help="Directory to save the results", required=True
    )
    parser.add_argument(
        "--scratch_dir", type=Path, help="Scratch directory to store temporary files", required=True
    )
    args = parser.parse_args(command_line_args)

    return args


def xyz_str_to_coords(xyz_str):
    """
    Convert a string of xyz format to a numpy array of coordinates
    """
    xyz_lines = xyz_str.split("\n")[2:]
    coords = [
        [float(coord) for coord in line.split()[1:]] for line in xyz_lines if line
    ]
    coords = np.array(coords)
    atomic_numbers = [get_element(line.split()[0]).number for line in xyz_lines if line]
    return atomic_numbers, coords
