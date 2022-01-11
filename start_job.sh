#!/bin/bash
#SBATCH --job-name="$ALGORITHM-$BENCHMARK-$EXAMPLES-$NUM_MODELS"
#SBATCH --partition=gpu_8
#SBATCH --nodes=5
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --parsable

EPOCHS=60000
EPOCHS_TO_STORE=60001
EPOCHS_TO_TEST=5000

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    export "$KEY"=$VALUE
done

python train.py --algorithm $ALGORITHM \
                --wandb True \
                --num_epochs $EPOCHS \
                --benchmark $BENCHMARK \
                --num_models $NUM_MODELS \
                --k_shot $NUM_SAMPLES \
                --epochs_to_store $EPOCHS_TO_STORE \
                --epochs_to_test $EPOCHS_TO_TEST
