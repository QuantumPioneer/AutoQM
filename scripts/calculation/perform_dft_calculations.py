import os
import pandas as pd
from rdkit import Chem
from argparse import ArgumentParser

from rdmc.mol import RDKitMol

from radical_workflow.calculation.dft_calculation import dft_scf_opt

parser = ArgumentParser()
parser.add_argument(
    "--input_smiles",
    type=str,
    required=True,
    help="input smiles included in a .csv file",
)
parser.add_argument(
    "--input_geometry",
    type=str,
    required=True,
    help="input geometry included in a .sdf file",
)
parser.add_argument(
    "--output_folder", type=str, default="output", help="output folder name"
)
parser.add_argument(
    "--task_id",
    type=int,
    default=0,
    help="task id for the calculation",
)
parser.add_argument(
    "--num_tasks",
    type=int,
    default=1,
    help="number of tasks for the calculation",
)

# DFT optimization and frequency calculation
parser.add_argument(
    "--DFT_opt_freq_folder",
    type=str,
    default="DFT_opt_freq",
    help="folder for DFT optimization and frequency calculation",
)
parser.add_argument(
    "--DFT_opt_freq_theory",
    type=str,
    default="#P opt=(ts,calcall,maxcycle=32,noeig,nomicro,cartesian) scf=(xqc) iop(7/33=1) iop(2/9=2000) guess=mix wb97xd/def2svp",
    help="level of theory for the DFT calculation",
)
parser.add_argument(
    "--DFT_opt_freq_n_procs",
    type=int,
    default=16,
    help="number of process for DFT calculations",
)
parser.add_argument(
    "--DFT_opt_freq_job_ram",
    type=int,
    default=62400,  # 3900*16
    help="amount of ram (MB) allocated for each DFT calculation",
)

# specify paths
parser.add_argument(
    "--XTB_path", type=str, required=False, default=None, help="path to installed XTB"
)
parser.add_argument(
    "--G16_path",
    type=str,
    required=False,
    default=None,
    help="path to installed Gaussian 16",
)
parser.add_argument(
    "--RDMC_path",
    type=str,
    required=False,
    default=None,
    help="path to RDMC to use xtb-gaussian script for xtb optimization calculation.",
)
parser.add_argument(
    "--COSMOtherm_path",
    type=str,
    required=False,
    default=None,
    help="path to COSMOthermo",
)
parser.add_argument(
    "--COSMO_database_path",
    type=str,
    required=False,
    default=None,
    help="path to COSMO_database",
)
parser.add_argument(
    "--ORCA_path", type=str, required=False, default=None, help="path to ORCA"
)

parser.add_argument("--scratch_dir", type=str, required=True, help="scratch directory")

args = parser.parse_args()

XTB_PATH = args.XTB_path
G16_PATH = args.G16_path
RDMC_PATH = args.RDMC_path
COSMOTHERM_PATH = args.COSMOtherm_path
COSMO_DATABASE_PATH = args.COSMO_database_path
ORCA_PATH = args.ORCA_path

submit_dir = os.path.abspath(os.getcwd())
output_dir = os.path.join(submit_dir, args.output_folder)

df = pd.read_csv(args.input_smiles)
assert len(df["id"]) == len(set(df["id"])), "ids must be unique"

assert (
    G16_PATH is not None
), f"G16_PATH must be provided for DFT optimization and frequency calculation"
assert RDMC_PATH is not None, f"RDMC_PATH must be provided to read sdf files"

# create id to smile mapping
mol_ids = df["id"].tolist()
smiles_list = df["rsmi"].tolist()
mol_id_to_smi = dict(zip(mol_ids, smiles_list))
mol_id_to_charge = dict()
mol_id_to_mult = dict()
for k, v in mol_id_to_smi.items():
    try:
        mol = Chem.MolFromSmiles(v)
    except Exception as e:
        print(f"Cannot translate smi {v} to molecule for species {k}")

    try:
        charge = Chem.GetFormalCharge(mol)
        mol_id_to_charge[k] = charge
    except Exception as e:
        print(f"Cannot determine molecular charge for species {k} with smi {v}")

    num_radical_elec = 0
    for atom in mol.GetAtoms():
        num_radical_elec += atom.GetNumRadicalElectrons()
    mol_id_to_mult[k] = num_radical_elec + 1

os.makedirs(args.scratch_dir, exist_ok=True)
mol_ids_smis = list(zip(mol_ids, smiles_list))

# create id to xyz mapping
mols = RDKitMol.FromFile(args.input_geometry, removeHs=False, sanitize=False)
mol_id_to_xyz = dict()
for mol in mols:
    mol_id = int(mol.GetProp("_Name").split("_")[0])
    xyz = mol.ToXYZ(header=False)
    mol_id_to_xyz[mol_id] = xyz

print("DFT TS opt & freq")

for _ in range(1):
    print("Making input files for TS DFT optimization and frequency calculation")

    DFT_opt_freq_dir = os.path.join(output_dir, args.DFT_opt_freq_folder)
    os.makedirs(DFT_opt_freq_dir, exist_ok=True)
    inputs_dir = os.path.join(DFT_opt_freq_dir, "inputs")
    outputs_dir = os.path.join(DFT_opt_freq_dir, "outputs")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    for mol_id, smi in mol_ids_smis[args.task_id : len(mol_ids_smis) : args.num_tasks]:
        ids = mol_id // 1000
        subinputs_dir = os.path.join(DFT_opt_freq_dir, "inputs", f"inputs_{ids}")
        suboutputs_dir = os.path.join(DFT_opt_freq_dir, "outputs", f"outputs_{ids}")
        os.makedirs(suboutputs_dir, exist_ok=True)

        if not os.path.exists(os.path.join(suboutputs_dir, f"{mol_id}.log")):
            if mol_id in mol_id_to_xyz:
                os.makedirs(subinputs_dir, exist_ok=True)
                if not os.path.exists(
                    os.path.join(subinputs_dir, f"{mol_id}.in")
                ) and not os.path.exists(os.path.join(subinputs_dir, f"{mol_id}.tmp")):
                    with open(os.path.join(subinputs_dir, f"{mol_id}.in"), "w") as f:
                        f.write(mol_id)
                    print(mol_id)

    print("Optimizing lowest energy semiempirical opted conformer with DFT method...")

    DFT_opt_freq_theories = [args.DFT_opt_freq_theory]

    for _ in range(1):
        for subinputs_folder in os.listdir(os.path.join(DFT_opt_freq_dir, "inputs")):
            ids = subinputs_folder.split("_")[1]
            subinputs_dir = os.path.join(DFT_opt_freq_dir, "inputs", f"inputs_{ids}")
            suboutputs_dir = os.path.join(DFT_opt_freq_dir, "outputs", f"outputs_{ids}")
            for input_file in os.listdir(subinputs_dir):
                if ".in" in input_file:
                    mol_id = input_file.split(".in")[0]
                    try:
                        os.rename(
                            os.path.join(subinputs_dir, input_file),
                            os.path.join(subinputs_dir, f"{mol_id}.tmp"),
                        )
                    except:
                        continue
                    else:
                        ids = mol_id // 1000
                        smi = mol_id_to_smi[mol_id]
                        charge = mol_id_to_charge[mol_id]
                        mult = mol_id_to_mult[mol_id]
                        print(mol_id)
                        print(smi)

                        mol_id_to_semiempirical_opted_xyz = dict()
                        mol_id_to_semiempirical_opted_xyz[mol_id] = mol_id_to_xyz[
                            mol_id
                        ]

                        converged = dft_scf_opt(
                            mol_id,
                            smi,
                            mol_id_to_semiempirical_opted_xyz,
                            G16_PATH,
                            DFT_opt_freq_theories,
                            args.DFT_opt_freq_n_procs,
                            args.DFT_opt_freq_job_ram,
                            charge,
                            mult,
                            args.scratch_dir,
                            suboutputs_dir,
                            subinputs_dir,
                        )

                        if converged:
                            os.remove(os.path.join(subinputs_dir, f"{mol_id}.tmp"))

    print("DFT optimization and frequency calculation done.")

print("Done!")
