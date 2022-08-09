#!/bin/sh
# for un_corpus but replace with `mtnt` for mtnt experiments

# centralized
python main_lm.py --data un_mt_corpus --model facebook/m2m100_418M --n_cpus 1 --n_gpus 2 --batch_size 2 --batch_accum 16 --lang_mix 0.99 --centralized --n_iterations 50 --lr 5e-5

# IID FL
python main_lm.py --data un_mt_corpus --model facebook/m2m100_418M --n_cpus 1 --n_gpus 3 --batch_size 2 --batch_accum 16 --lang_mix 0.99 --n_iterations 50 --lr 5e-5

# non-IID FL
python main_lm.py --data un_mt_corpus --model facebook/m2m100_418M --n_cpus 1 --n_gpus 3 --batch_size 2 --batch_accum 16 --lang_mix 0.0 --n_iterations 50 --lr 5e-5

# For eval add "--n_iterations 0 --load_model <PATH_TO_MODEL.pt>"
# For random initialization add "--random_init" to the model (garbage results though, due to the small data)
