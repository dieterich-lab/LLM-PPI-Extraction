#!/bin/bash

#SBATCH --gres=gpu:turing:4
#SBATCH --job-name=marker_parse
#SBATCH --output=../outputs/slurm/marker_parse.txt
#SBATCH --partition=gpu
#SBATCH --mem=50G


# Template for SLURM GPU handling scripts
# From https://techfak.net/gpu-cluster

################################################################################################
echo "==== Start of GPU information ===="

CUDA_DEVICE=$(echo "$CUDA_VISIBLE_DEVICES," | cut -d',' -f $((SLURM_LOCALID + 1)))
T_REGEX='^[0-9]$'
if ! [[ "$CUDA_DEVICE" =~ $T_REGEX ]]; then
        echo "error no reserved gpu provided"
        exit 1
fi

# Print debug information

echo -e "SLURM job:\t$SLURM_JOBID"
echo -e "CUDA_VISIBLE_DEVICES:\t$CUDA_VISIBLE_DEVICES"
echo "Device list:"
nvidia-smi --query-gpu=name,gpu_uuid --format=csv -i "$CUDA_VISIBLE_DEVICES" | tail -n +2
NGPUS=$(nvidia-smi --query-gpu=name,gpu_uuid --format=csv -i "$CUDA_VISIBLE_DEVICES" | tail -n +2 | wc -l)
# echo "$(nvidia-smi --query-gpu=name,gpu_uuid --format=csv -i $CUDA_VISIBLE_DEVICES | tail -n +2)"
echo "==== End of GPU information ===="
echo ""
#################################################################################################

. ~/.venvs/linda/bin/activate
python -u marker_parse.py
