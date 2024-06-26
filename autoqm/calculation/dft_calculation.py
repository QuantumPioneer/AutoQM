import logging

logging.basicConfig(level=logging.INFO)
import shutil
from rdkit import Chem
import copy
import csv
import os
import subprocess
import numpy as np
import time
from pathlib import Path

from .file_parser import mol2xyz, xyz2com, clean_xyz_str
from .grab_QM_descriptors import read_log
from .log_parser import G16Log
from autoqm.parser.dft_opt_freq_parser import read_log_file, check_job_status


def dft_scf_qm_descriptor(
    g16_path,
    job_id,
    xyz_str,
    charge,
    mult,
    template,
    scratch_dir,
    subinputs_dir,
    suboutputs_dir,
):

    current_dir = Path.cwd()

    job_scratch_dir = scratch_dir / f"{job_id}"
    logging.info(f"Creating scratch directory {job_scratch_dir}...")
    job_scratch_dir.mkdir()

    subprocess.run(
        f"export GAUSS_SCRDIR={job_scratch_dir.absolute()}", shell=True, check=True
    )

    job_tmp_output_dir = suboutputs_dir / f"{job_id}"
    try:
        shutil.rmtree(job_tmp_output_dir, ignore_errors=True)
    except FileNotFoundError:
        pass
    try:
        job_tmp_output_dir.mkdir()
    except FileExistsError:
        logging.error(f"{job_tmp_output_dir} exists. Assuming another worker is using it. Skipping...")
        shutil.rmtree(job_scratch_dir, ignore_errors=True)
        return 

    os.chdir(job_tmp_output_dir)

    g16_command = os.path.join(g16_path, "g16")
    xyz_str = clean_xyz_str(xyz_str)
    content = template.format(job_id=job_id, charge=charge, mult=mult, xyz_str=xyz_str)

    comfile = Path(f"{job_id}.gjf").absolute()
    logfile = Path(f"{job_id}.log").absolute()
    outfile = Path(f"{job_id}.out").absolute()

    with open(comfile, "w") as f:
        f.write(content)

    start_time = time.time()
    with open(outfile, "w") as out:
        subprocess.run(
            "{} < {} >> {}".format(g16_command, comfile, logfile),
            shell=True,
            stdout=out,
            stderr=out,
        )
    end_time = time.time()

    logging.info(f"Optimization of {job_id} took {end_time - start_time} seconds.")

    with open(logfile, "r") as f:
        lines = f.readlines()[-10:]
        if any("Normal termination" in line for line in lines):

            new_logfile = suboutputs_dir / f"{job_id}.log"
            shutil.copyfile(logfile, new_logfile)
            logging.info(f"Normal temrination for {new_logfile}")

            os.chdir(current_dir)
            shutil.rmtree(job_tmp_output_dir)
            job_tmp_input_path = subinputs_dir / f"{job_id}.tmp"
            job_tmp_input_path.unlink()
            shutil.rmtree(job_scratch_dir)

        else:

            logging.error(f"Abnormal temrination for {logfile}:")
            for line in lines:
                logging.error(line)

            os.chdir(current_dir)
            job_tmp_input_path = subinputs_dir / f"{job_id}.tmp"
            job_tmp_input_path.unlink()
            shutil.rmtree(job_scratch_dir)


def dft_scf_opt(
    job_id,
    job_xyz,
    g16_path,
    level_of_theory,
    n_procs,
    job_ram,
    charge,
    mult,
    scratch_dir,
    subinputs_dir,
    suboutputs_dir,
):
    current_dir = os.getcwd()

    job_scratch_dir = os.path.join(scratch_dir, f"{job_id}")
    print(job_scratch_dir)

    os.makedirs(job_scratch_dir)
    os.chdir(job_scratch_dir)

    g16_command = os.path.join(g16_path, "g16")
    head = "%chk={}.chk\n%nprocshared={}\n%mem={}mb\n{}\n".format(
        job_id, n_procs, job_ram, level_of_theory
    )

    comfile = f"{job_id}.gjf"
    xyz2com(job_xyz, head=head, comfile=comfile, charge=charge, mult=mult, footer="\n")
    shutil.copyfile(comfile, os.path.join(suboutputs_dir, f"{job_id}.gjf"))

    logfile = f"{job_id}.log"
    outfile = f"{job_id}.out"

    start_time = time.time()
    with open(outfile, "w") as out:
        subprocess.run(
            "{} < {} >> {}".format(g16_command, comfile, logfile),
            shell=True,
            stdout=out,
            stderr=out,
        )
    end_time = time.time()
    print(
        f"Optimization of {job_id} with {level_of_theory} took {end_time - start_time} seconds."
    )

    shutil.copyfile(logfile, os.path.join(suboutputs_dir, f"{job_id}.log"))
    try:
        os.remove(os.path.join(subinputs_dir, f"{job_id}.tmp"))
    except FileNotFoundError:
        print(
            f"{os.path.join(subinputs_dir, f'{job_id}.tmp')} not found. Already deleted?"
        )

    job_stat = check_job_status(read_log_file(logfile))

    os.chdir(current_dir)
    shutil.rmtree(job_scratch_dir)

    return job_stat


def dft_scf_sp(
    job_id, g16_path, level_of_theory, n_procs, logger, job_ram, charge, mult
):
    sdf = job_id + ".sdf"

    mol = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False)[0]
    job_xyz = mol2xyz(mol)

    g16_command = os.path.join(g16_path, "g16")
    head = "%chk={}.chk\n%nprocshared={}\n%mem={}mb\n{}\n".format(
        job_id, n_procs, job_ram, level_of_theory
    )

    comfile = job_id + ".gjf"
    xyz2com(job_xyz, head=head, comfile=comfile, charge=charge, mult=mult, footer="\n")

    logfile = job_id + ".log"
    outfile = job_id + ".out"
    with open(outfile, "w") as out:
        subprocess.run(
            "{} < {} >> {}".format(g16_command, comfile, logfile),
            shell=True,
            stdout=out,
            stderr=out,
        )

    os.remove(sdf)


def save_dft_sp_results(
    folder, done_jobs_record, task_id, mol_id_to_smi_dict, semiempirical_methods
):
    """extract dft single point calculation results for geometry optimized with different semiempirical methods"""
    result_file_path = f"dft_sp_result_{task_id}.csv"
    header = ["id", "smiles", "GFN2-XTB_opt_DFT_sp", "am1_opt_DFT_sp", "pm7_opt_DFT_sp"]
    with open(result_file_path, "w") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the header
        csvwriter.writerow(header)

        for job_id, semiempirical_methods in done_jobs_record.test_DFT_sp.items():
            each_data_list = [job_id, mol_id_to_smi_dict[job_id]]
            for semiempirical_method in semiempirical_methods:
                log_file_path = os.path.join(
                    folder, job_id, semiempirical_method, job_id + ".log"
                )
                g16log = G16Log(log_file_path)
                en = g16log.E
                each_data_list.append(str(en))
            csvwriter.writerow(each_data_list)
