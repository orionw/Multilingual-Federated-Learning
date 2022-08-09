#!/bin/sh

# centralized
python main_lm.py --data nc --model xlm-roberta-base --n_cpus 1 --n_gpus 2 --batch_size 8 --batch_accum 4 --lang_mix 0.99 --centralized --n_iterations 10 --lr 1e-5 

# IID FL
python main_lm.py --data nc --model xlm-roberta-base --n_cpus 1 --n_gpus 6 --batch_size 8 --batch_accum 4 --lang_mix 0.99 --n_iterations 10 --lr 1e-5 

# Non-IID FL
python main_lm.py --data nc --model xlm-roberta-base --n_cpus 1 --n_gpus 6 --batch_size 8 --batch_accum 4 --lang_mix 0.0 --n_iterations 10 --lr 1e-5 

# For eval add "--n_iterations 0 --load_model <PATH_TO_MODEL.pt>"
# For random initialization add "--random_init" to the model and change n_iterations to 50.
