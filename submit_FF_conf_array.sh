#!/bin/bash -l
#SBATCH -J FF_input
#SBATCH -n 1
#SBATCH --array=0-31

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

conda activate rdmc_env

#xtb
source /home/gridsan/groups/RMG/Software/xtb-6.4.1/share/xtb/config_env.bash
XTB_PATH=/home/gridsan/groups/RMG/Software/xtb-6.4.1/bin
export PATH=$XTB_PATH:$PATH

#QMD
QMD_PATH=/home/gridsan/groups/RMG/Software/QM_descriptors_calculation-radical_workflow
input_smiles=/home/gridsan/groups/RMG/Projects/Hao-Wei-Oscar-Yunsie/production_run/HAbs/inputs/TS_cosmo_dlpno/wb97xd_and_xtb_opted_ts_combo_results_hashed_sep1a_ts_input.csv

scratch_dir=$HOME/scratch/$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
mkdir -p $scratch_dir

python $QMD_PATH/main_FF_conf.py --input_smiles $input_smiles --XTB_path $XTB_PATH --scratch_dir $scratch_dir