#!/bin/sh
# for un_corpus but replace with `wmt` for europarl experiments

# centralized
python main_lm.py --data un_corpus --model distilbert-base-multilingual-cased --n_cpus 1 --n_gpus 2 --batch_size 10 --batch_accum 3 --lang_mix 0.99 --centralized --n_iterations 100 --lr 5e-5

# IID FL
python main_lm.py --data un_corpus --model distilbert-base-multilingual-cased --n_cpus 1 --n_gpus 5 --batch_size 10 --batch_accum 3 --lang_mix 0.99 --n_iterations 100 --lr 5e-5

# non-IID FL
python main_lm.py --data un_corpus --model distilbert-base-multilingual-cased --n_cpus 1 --n_gpus 5 --batch_size 10 --batch_accum 3 --lang_mix 0.0 --n_iterations 100 --lr 5e-5

# For eval add "--n_iterations 0 --load_model <PATH_TO_MODEL.pt>"
# For random initialization add "--random_init" to the model and double n_iterations
