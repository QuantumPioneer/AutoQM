from argparse import ArgumentParser
import os
import shutil
import time
import tarfile
import csv

import pickle as pkl
import pandas as pd
import traceback

from rdkit import Chem

from radical_workflow.calculation.wft_calculation import generate_dlpno_sp_input
from radical_workflow.calculation.cosmo_calculation import cosmo_calc

parser = ArgumentParser()
parser.add_argument('--input_smiles', type=str, required=True,
                    help='input smiles included in a .csv file')
parser.add_argument('--output_folder', type=str, default='output',
                    help='output folder name')
parser.add_argument('--scratch_dir', type=str, required=True,
                    help='scfratch directory')
parser.add_argument('--task_id', type=int, default=0,
                    help='task id for the calculation',)
parser.add_argument('--num_tasks', type=int, default=1,
                    help='number of tasks for the calculation',)

# Turbomole and COSMO calculation
parser.add_argument('--COSMO_folder', type=str, default='COSMO_calc',
                    help='folder for COSMO calculation',)
parser.add_argument('--COSMO_temperatures', type=str, nargs="+", required=False, default=['297.15', '298.15', '299.15'],
                    help='temperatures used for COSMO calculation')
parser.add_argument('--COSMO_input_pure_solvents', type=str, required=False, default='common_solvent_list_final.csv',
                    help='input file containing pure solvents used for COSMO calculation.')

# specify paths
parser.add_argument('--XTB_path', type=str, required=False, default=None,
                    help='path to installed XTB')
parser.add_argument('--G16_path', type=str, required=False, default=None,
                    help='path to installed Gaussian 16')
parser.add_argument('--RDMC_path', type=str, required=False, default=None,
                    help='path to RDMC to use xtb-gaussian script for xtb optimization calculation.')
parser.add_argument('--COSMOtherm_path', type=str, required=False, default=None,
                    help='path to COSMOthermo')
parser.add_argument('--COSMO_database_path', type=str, required=False, default=None,
                    help='path to COSMO_database')
parser.add_argument('--ORCA_path', type=str, required=False, default=None,
                    help='path to ORCA')

args = parser.parse_args()

# input files
df = pd.read_csv(args.input_smiles, index_col=0)

if "smiles" in df.columns:
    mol_smis = list(df.smiles)
elif "rxn_smi" in df.columns:
    mol_smis = list(df.rxn_smi)
else:
    raise ValueError("Cannot find smiles or rxn_smi in input file.")

mol_id_to_smi_dict = dict(zip(df.id, mol_smis))
mol_id_to_charge_dict = dict()
mol_id_to_mult_dict = dict()
for k, v in mol_id_to_smi_dict.items():
    try:
        mol = Chem.MolFromSmiles(v)
    except Exception as e:
        print(f'Cannot translate smi {v} to molecule for species {k}')

    try:
        charge = Chem.GetFormalCharge(mol)
        mol_id_to_charge_dict[k] = charge
    except Exception as e:
        print(f'Cannot determine molecular charge for species {k} with smi {v}')

    num_radical_elec = 0
    for atom in mol.GetAtoms():
        num_radical_elec += atom.GetNumRadicalElectrons()
    mol_id_to_mult_dict[k] =  num_radical_elec + 1

submit_dir = os.path.abspath(os.getcwd())
project_dir = os.path.abspath(os.path.join(args.output_folder))
COSMO_dir = os.path.join(project_dir, args.COSMO_folder)

df_pure = pd.read_csv(os.path.join(submit_dir,args.COSMO_input_pure_solvents))
df_pure = df_pure.reset_index()
COSMOTHERM_PATH = args.COSMOtherm_path
COSMO_DATABASE_PATH = args.COSMO_database_path
assert COSMOTHERM_PATH is not None and COSMO_DATABASE_PATH is not None, "COSMOTHERM_PATH and COSMO_DATABASE_PATH must be provided for COSMO calc"

print("Making inputs and outputs dir...")
mol_ids = list(df["id"])
smiles_list = list(df["smiles"])
inputs_dir = os.path.join(COSMO_dir, "inputs")
os.makedirs(inputs_dir, exist_ok=True)
outputs_dir = os.path.join(COSMO_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

print("Making helper input files...")

mol_ids_smis = list(zip(mol_ids, smiles_list))
for mol_id, smi in mol_ids_smis[args.task_id::args.num_tasks]:
    if mol_id in xyz_DFT_opt_dict:
        ids = str(int(int(mol_id.split("id")[1])/1000))
        subinputs_dir = os.path.join(inputs_dir, f"inputs_{ids}")
        suboutputs_dir = os.path.join(outputs_dir, f"outputs_{ids}")
        os.makedirs(suboutputs_dir, exist_ok=True)
        mol_id_path = os.path.join(subinputs_dir, f"{mol_id}.in")
        tmp_mol_id_path = os.path.join(subinputs_dir, f"{mol_id}.tmp")
        if not os.path.exists(os.path.join(suboutputs_dir, f"{mol_id}.tar")):
            os.makedirs(subinputs_dir, exist_ok=True)
            if not os.path.exists(mol_id_path) and not os.path.exists(tmp_mol_id_path):
                with open(mol_id_path, "w+") as f:
                    f.write(mol_id)
                print(mol_id)

print("Starting COSMO calculations...")
for _ in range(5):
    for subinputs_folder in os.listdir(os.path.join(COSMO_dir, "inputs")):
        ids = subinputs_folder.split("_")[1]
        subinputs_dir = os.path.join(COSMO_dir, "inputs", subinputs_folder)
        suboutputs_dir = os.path.join(COSMO_dir, "outputs", f"outputs_{ids}")
        for input_file in os.listdir(subinputs_dir):
            if ".in" in input_file:
                mol_id = input_file.split(".in")[0]
                try:
                    os.rename(os.path.join(subinputs_dir, input_file), os.path.join(subinputs_dir, f"{mol_id}.tmp"))
                except:
                    continue
                else:
                    print(mol_id)
                    ids = str(int(int(mol_id.split("id")[1])/1000))
                    charge = mol_id_to_charge_dict[mol_id]
                    mult = mol_id_to_mult_dict[mol_id]
                    coords = xyz_DFT_opt_dict[mol_id]
                    tmp_mol_dir = os.path.join(suboutputs_dir, mol_id)
                    os.makedirs(tmp_mol_dir, exist_ok=True)
                    cosmo_calc(mol_id, COSMOTHERM_PATH, COSMO_DATABASE_PATH, charge, mult, args.COSMO_temperatures, df_pure, coords, args.scratch_dir, tmp_mol_dir, suboutputs_dir, subinputs_dir)

print("Done!")