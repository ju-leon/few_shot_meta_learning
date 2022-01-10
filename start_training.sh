#!/bin/bash

EPOCHS=100
EXAMPLES=4
BENCHMARK=Sinusoid1D

python train.py --algorithm maml --wandb True --num_epochs $EPOCHS --benchmark $BENCHMARK --num_example_tasks $EXAMPLES
python train.py --algorithm platipus --wandb True --num_epochs $EPOCHS --benchmark $BENCHMARK --num_example_tasks $EXAMPLES
python train.py --algorithm bmaml --wandb True --num_epochs $EPOCHS --benchmark $BENCHMARK --num_example_tasks $EXAMPLES