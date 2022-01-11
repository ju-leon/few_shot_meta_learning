#!/bin/bash

module load devel/cuda/11.0
module load devel/cudnn/10.2

for num_samples in 1 2 4 8
do
    for benchmark in Sinusiod Affine SinusAffine
    do
        sh start_job.sh ALGORITHM="maml" NUM_SAMPLES=$num_samples BENCHMARK=$benchmark NUM_MODELS=1 
        for num_models in 5 100 1000
        do
            sh start_job.sh ALGORITHM="platipus" NUM_SAMPLES=$num_samples BENCHMARK=$benchmark NUM_MODELS=$num_models
            sh start_job.sh ALGORITHM="bmaml" NUM_SAMPLES=$num_samples BENCHMARK=$benchmark NUM_MODELS=$num_models
        done
    done
done