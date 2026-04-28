#!/bin/bash
#SBATCH --nodes=5
#SBATCH --job-name=cvs
#SBATCH --time=10:00:00
#SBATCH --partition=pdebug
#SBATCH --error=cv.txt
#SBATCH --output=cv.txt

mkdir -p logs
source /g/g90/dhakal1/.venv_cv_gpu/bin/activate

python main1_4k.py \
    --skip-download \
    --xlsx pku_aigiqa_4k/images/annotation.xlsx \
    --epochs 50 \
    --batch 32 \
    --workers 8