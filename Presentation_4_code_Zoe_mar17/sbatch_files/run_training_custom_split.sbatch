#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=40:00:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=MIL_train              # Job name (no spaces)
#SBATCH --output=patch_order_multirun.out
#SBATCH --error=patch_order_multirun.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=zel9906@northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /home/zel9906/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /home/zel9906/STAT390_Team4/Presentation_4_code_Zoe_mar17/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

cd /home/zel9906/STAT390_Team4/Presentation_4_code_Zoe_mar17/

# without loading checkpoints, trying 5 diff train / test splits

python3 main.py \
    --multi_run \
    --splits_dir /home/zel9906/STAT390_Team4/Presentation_4_code_Zoe_mar17/no_train_val_leakage_splits \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --aggregate_output_dir /home/zel9906/STAT390_Team4/Presentation_4_code_Zoe_mar17/patch_order_results \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 
 # no attention or load splits needed for this run