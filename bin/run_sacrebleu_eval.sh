#!/bin/sh
# How to evaluate sacrebleu on output translations
sacrebleu <Gold_Translation> -i <Output_Translation> -m bleu -b -w 1 --confidence
# If using ja add `--tokenize ja-mecab` and if using zh add `--tokenize zh`