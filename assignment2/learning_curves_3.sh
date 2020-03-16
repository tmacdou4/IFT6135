#!/bin/bash

# ## PROBLEM-SPECIFIC INSTRUCTIONS:
# - For Problem 3.1 the hyperparameter settings you should run are as follows

# - For Problem 3.2 the hyperparameter settings you should run are as follows

# - For Problem 3.3 the hyperparameter settings you should run are as follows

# - For Problem 3.4 the hyperparameter settings you should run are as follows
python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.6 --num_epochs=20
python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=1024 --num_layers=6 --dp_keep_prob=0.9 --num_epochs=20
