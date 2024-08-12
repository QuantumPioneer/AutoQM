"""
This module computes rate coefficient from .csv file containing energies and frequencies
"""

import ast
import logging

from copy import deepcopy
import pandas as pd
from joblib import Parallel, delayed

from arkane.kinetics import KineticsJob
from rmgpy import constants
from rmgpy.kinetics.tunneling import Eckart
from rmgpy.molecule import Molecule
from rmgpy.molecule.element import get_element
from rmgpy.reaction import Reaction
from rmgpy.species import Species, TransitionState
from utils import (
    get_lot_and_freq_scale,
    get_rmg_conformer,
    parse_command_line_arguments,
    xyz_str_to_coords,
)

# get logger
logger = logging.getLogger()


def get_rmg_conformer_from_df(
    row,
    freq_scale,
    energy_level,
    freq_level,
    energy_software,
    freq_software,
    use_atom_corrections,
    use_bond_corrections,
    bac_type,
    scr_dir=None,
):
    # No bond correction at this moment

    level_of_theory, freq_scale = get_lot_and_freq_scale(
        energy_level=energy_level,
        freq_level=freq_level,
        energy_software=energy_software,
        freq_software=freq_software,
        freq_scale=freq_scale,
    )

    rmg_conformers = []

    for spc_label in ["ts", "r1", "r2", "p1", "p2"]:


        if spc_label == "ts":
            molecule = None
            multiplicity = row["multiplicity"]
            use_bond_corrections = False
        else:
            molecule = Molecule().from_smiles(row[f"{spc_label}smi"])
            multiplicity = molecule.multiplicity

        if spc_label == "ts":
            xyz = row["std_xyz_str"]
        else:
            xyz = row[f"{spc_label}_matched_std_xyz_str"]
        atomic_numbers, coords = xyz_str_to_coords(xyz)
        mass = (
            sum(
                [
                    get_element(int(atomic_number)).mass
                    for atomic_number in atomic_numbers
                ]
            )
            / constants.Na,
            "kg",
        )

        frequencies = ast.literal_eval(row[f"{spc_label}_dft_frequencies"])

        if energy_level == "qgdlpnoccsd(t)f12d/ccpvtzf12":
            e_electronic = row[f"{spc_label}_dlpno_sp_hartree"]
        elif energy_level == "qgwb97xd/def2svp":
            e_electronic = row[f"{spc_label}_dft_hartreefock_energy_hartree"]
        else:
            raise ValueError(f"Energy level {energy_level} not recognized")

        if isinstance(e_electronic, str):
            e_electronic = float(e_electronic)

        e_electronic = e_electronic * 2625500  # hartree to J/mol

        try:
            rmg_conformer = get_rmg_conformer(
                label=spc_label,
                level_of_theory=level_of_theory,
                e_electronic=e_electronic,
                frequencies=frequencies,
                coords=coords,
                numbers=atomic_numbers,
                mass=mass,
                multiplicity=multiplicity,
                freq_scale=freq_scale,
                molecule=molecule,
                use_atom_corrections=use_atom_corrections,
                use_bond_corrections=use_bond_corrections,
                bac_type=bac_type,
                scr_dir=scr_dir,
            )
            rmg_conformers.append(rmg_conformer)
        except Exception as e:
            logger.error(f"Error in getting rmg conformer for {row.values}: {e}")
            return None

    return rmg_conformers


def calc_rate_coefficient(
    row,
    freq_scale,
    energy_level,
    freq_level,
    energy_software,
    freq_software,
    Tmin=None,
    Tmax=None,
    Tcount=0,
    Tlist=None,
    sensitivity_conditions=None,
    three_params=True,
    scr_dir=None,
):
    
    if energy_level == "qgwb97xd/def2svp":
        kinetics = None

    elif energy_level == "qgdlpnoccsd(t)f12d/ccpvtzf12":
        # get rates
        output = get_rmg_conformer_from_df(
            row,
            freq_scale,
            energy_level,
            freq_level,
            energy_software=energy_software,
            freq_software=freq_software,
            use_atom_corrections=False,
            use_bond_corrections=False,
            bac_type="p",
            scr_dir=scr_dir,
        )

        if output is None:
            kinetics = None

        else:

            ts, r1, r2, p1, p2 = output

            spc_r1 = Species().from_smiles(row["r1smi"])
            spc_r1.conformer = r1
            spc_r2 = Species().from_smiles(row["r2smi"])
            spc_r2.conformer = r2
            spc_p1 = Species().from_smiles(row["p1smi"])
            spc_p1.conformer = p1
            spc_p2 = Species().from_smiles(row["p2smi"])
            spc_p2.conformer = p2

            neg_frequency = row["neg_freq"] * freq_scale
            neg_frequency = (neg_frequency, "cm^-1")

            E0_reactants  =  (r1.E0.value + r2.E0.value, "kJ/mol")
            E0_products  =  (p1.E0.value + p2.E0.value, "kJ/mol")
            E0_transitionstate  =  ts.E0
            
            spc_ts = TransitionState(
                conformer=ts,
                frequency=neg_frequency,
                tunneling=Eckart(frequency=neg_frequency, E0_reac=E0_reactants, E0_TS=E0_transitionstate, E0_prod=E0_products),
            )

            rxn = Reaction(
                reactants=[spc_r1, spc_r2], products=[spc_p1, spc_p2], transition_state=spc_ts
            )

            kinetics_job = KineticsJob(
                reaction=rxn,
                Tmin=Tmin,
                Tmax=Tmax,
                Tcount=Tcount,
                Tlist=Tlist,
                sensitivity_conditions=sensitivity_conditions,
                three_params=three_params,
            )

            try:
                kinetics_job.generate_kinetics()
                kinetics = kinetics_job.reaction.kinetics
            except Exception as e:
                logger.error(f"Error in generating kinetics for {row}: {e}")
                kinetics = None

    else:
        raise ValueError(f"Energy level {energy_level} not recognized")
    
    # get reaction thermo (Petersson)
    output = get_rmg_conformer_from_df(
        row,
        freq_scale,
        energy_level,
        freq_level,
        energy_software=energy_software,
        freq_software=freq_software,
        use_atom_corrections=True,
        use_bond_corrections=True,
        bac_type="p",
        scr_dir=scr_dir,
    )

    if output is None:
        p_rxn = None

    else:

        _, r1, r2, p1, p2 = output

        spc_r1 = Species().from_smiles(row["r1smi"])
        spc_r1.conformer = r1
        spc_r2 = Species().from_smiles(row["r2smi"])
        spc_r2.conformer = r2
        spc_p1 = Species().from_smiles(row["p1smi"])
        spc_p1.conformer = p1
        spc_p2 = Species().from_smiles(row["p2smi"])
        spc_p2.conformer = p2

        p_rxn = Reaction(
            reactants=[spc_r1, spc_r2], products=[spc_p1, spc_p2], kinetics=deepcopy(kinetics)
        )

    # get reaction thermo (melius)
    output = get_rmg_conformer_from_df(
        row,
        freq_scale,
        energy_level,
        freq_level,
        energy_software=energy_software,
        freq_software=freq_software,
        use_atom_corrections=True,
        use_bond_corrections=True,
        bac_type="m",
        scr_dir=scr_dir,
    )

    if output is None:
        m_rxn = None

    else:

        _, r1, r2, p1, p2 = output

        spc_r1 = Species().from_smiles(row["r1smi"])
        spc_r1.conformer = r1
        spc_r2 = Species().from_smiles(row["r2smi"])
        spc_r2.conformer = r2
        spc_p1 = Species().from_smiles(row["p1smi"])
        spc_p1.conformer = p1
        spc_p2 = Species().from_smiles(row["p2smi"])
        spc_p2.conformer = p2

        m_rxn = Reaction(
            reactants=[spc_r1, spc_r2], products=[spc_p1, spc_p2], kinetics=deepcopy(kinetics)
        )

    return kinetics, p_rxn, m_rxn


def main():
    """
    The executable function
    """
    # 0. Parse input
    args = parse_command_line_arguments()

    df = pd.read_csv(args.csv_path)
    df = df.dropna()

    energy_level = args.energy_level
    freq_level = args.freq_level
    energy_software = args.energy_software
    freq_software = args.freq_software
    freq_scale = args.freq_scale

    reactions_list = Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(calc_rate_coefficient)(
            row,
            freq_scale,
            energy_level,
            freq_level,
            energy_software,
            freq_software,
            scr_dir=args.scratch_dir / f"reaction_{idx}",
        )
        for idx, row in df.iterrows()
    )

    def get_kinetics_values(kinetics, key):
        if kinetics is None:
            return None

        if key == "A":
            return kinetics.A.value_si
        elif key == "n":
            return kinetics.n.value_si
        elif key == "Ea":
            return kinetics.Ea.value_si
        else:
            raise ValueError(f"Key {key} is not recognized")

    def get_rxn_values(rxn, key):
        if rxn is None:
            return None

        if key == "deltaHrxn298":
            return rxn.get_enthalpy_of_reaction(298)
        elif key == "deltaGrxn298":
            return rxn.get_free_energy_of_reaction(298)
        else:
            raise ValueError(f"Key {key} is not recognized")
    
    def get_rev_kin(rxn):
        if rxn is None:
            return None
        
        if rxn.kinetics is None:
            return None
        
        return rxn.generate_reverse_rate_coefficient()

    kinetics_list = [x[0] for x in reactions_list]
    p_reaction_list = [x[1] for x in reactions_list]
    m_reaction_list = [x[2] for x in reactions_list]
    df["kinetics"] = kinetics_list
    df["p_reaction"] = p_reaction_list
    df["m_reaction"] = m_reaction_list
    df["p_rev_kinetics"] = [get_rev_kin(rxn) for rxn in p_reaction_list]
    df["m_rev_kinetics"] = [get_rev_kin(rxn) for rxn in m_reaction_list]

    for column in ["A", "n", "Ea"]:
        df[column] = df["kinetics"].apply(lambda x: get_kinetics_values(x, column))
        df["p_rev_" + column] = df["p_rev_kinetics"].apply(lambda x: get_kinetics_values(x, column))
        df["m_rev_" + column] = df["m_rev_kinetics"].apply(lambda x: get_kinetics_values(x, column))

    for column in ["deltaHrxn298", "deltaGrxn298"]:
        df["p_" + column] = df["p_reaction"].apply(lambda x: get_rxn_values(x, column))
        df["m_" + column] = df["m_reaction"].apply(lambda x: get_rxn_values(x, column))

    df = df.drop(columns=["kinetics", "p_reaction", "m_reaction", "p_rev_kinetics", "m_rev_kinetics"])

    df = df.dropna(subset=["A", "n", "Ea", "p_rev_A", "p_rev_n", "p_rev_Ea", "m_rev_A", "m_rev_n", "m_rev_Ea"])

    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
