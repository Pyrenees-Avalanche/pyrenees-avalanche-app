#!/bin/bash
#SBATCH --job-name=avalcd_swin

##SBATCH --partition=rtx6000
##SBATCH --gres=gpu:rtx6000:1
##SBATCH --cpus-per-task=6
##SBATCH --mem=32G

##SBATCH --partition=a40
##SBATCH --account=acct_gen
##SBATCH --qos=normal
##SBATCH --gres=gpu:a40:1
##SBATCH --cpus-per-task=6
##SBATCH --mem=32000M

#SBATCH --partition=l40s
#SBATCH --account=acct_gen
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M

##SBATCH --partition=titan
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
##SBATCH --mem=32000M

#SBATCH --time=08:00:00
#SBATCH --output=logs/avalcd_%j.out
#SBATCH --error=logs/avalcd_%j.err

set -e

PROJECT_ROOT="$HOME/pyrenees-avalanche-app"
cd "$PROJECT_ROOT/SAR/model"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate avalcd

echo "Running on $(hostname)"
nvidia-smi

STAGE="TEST"
DATA_ROOT="$PROJECT_ROOT/SAR/data/patches/test"
CHECKPOINT_PATH="$PROJECT_ROOT/SAR/data/checkpoints/avalcd_pretrained/best_model.pth"
EVENT_FOLDER="$PROJECT_ROOT/data/events/EVENT_CODE"

if [ "$STAGE" = "TRAIN" ]; then
    echo "Training..."

    python -u train.py \
        --dataset-root  $DATA_ROOT \
        --model swinunet \
        --model-size tiny \
        --fusion-type diff \
        --patch-size 128 \
        --batch-size 8 \
        --lr 1e-4 \
        --warmup-epochs 10 \
        --loss bce \
        --pos-weight 3.0 \
        --beta 1.0 \
        --precision-floor 0 \
        --description "CARE - SAR-only F1 optimized"

elif [ "$STAGE" = "TEST" ]; then
    echo "Testing..."

    python -u test.py \
        --dataset-root "$DATA_ROOT" \
        --model swinunet \
        --model-size tiny \
        --patch-size 128 \
        --model-ckpt "$CHECKPOINT_PATH" \
        --plot-every 10

elif [ "$STAGE" = "INFER" ]; then
    echo "Inferencing..."

    python -u infer.py \
        --event-path $EVENT_FOLDER \
        --model-ckpt $CHECKPOINT_PATH \
        --output-dir output/Tromso_20241220 \
        --patch-size 128 \
        --stride 64 \
        --blending gaussian

fi