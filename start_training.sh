#!/bin/bash

module load devel/cuda/11.0
module load devel/cudnn/10.2

for num_models in 5 100 1000
do
    sbatch start_job.sh ALGORITHM="platipus" NUM_MODELS=$num_models
    sbatch start_job.sh ALGORITHM="bmaml" NUM_MODELS=$num_models
done

sbatch start_job.sh ALGORITHM="maml" NUM_MODELS=1
