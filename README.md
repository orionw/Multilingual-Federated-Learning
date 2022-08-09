# Pretrained Models for Multilingual Federated Learning
Code and data setup for our paper: [Pretrained Models for Multilingual Federated Learning](https://aclanthology.org/2022.naacl-main.101/) by *Orion Weller, *Marc Marone, Vladimir Braverman, Dawn Lawrie, and Benjamin Van Durme. Many thanks to the great developers at the [flwr](https://flower.dev/) team who have prepared [excellent examples](https://github.com/adap/flower/tree/main/examples/simulation_pytorch).

## Enviroment Setup
NOTE: we used poetry following the advice of the flwr framework. 

0. Install poetry (`bash enviroment_setup/install_poetry.sh`)
1. Activate poetry (`bash enviroment_setup/activate_poetry.sh`)
2. Install dependecies (`poetry install`). NOTE: this takes a few minutes.

## Data Setup
0. After deciding which data setup you would like, look for the corresponding dataset in `create_data` For the sake of this readme, we will use the `mtnt` data.
1. `cd` into the folder (`cd create_data/make_mtnt_data`)
2. Follow the instructions in the `readme` located in the folder. It will typically have scripts for downloading, preprocessing, splitting, and then moving the data into the final location for the model.

## Training/Evaluating Federated Learning Models
0. Make sure the enviroment and the data have been set up as above.
1. Depending on the type of model you want to train (classification, LM, or MT) see the corresponding scripts in `bin/run_fl_{mt,tc,lm}.sh`. Each script contains information about how to run centralized, non-IID FL, or IID FL learning, as well as random initialization and/or evaluation.
2. To evaluate BLEU scores, be sure to install the sacrebleu script and evaluating using the format described in `bin/run_sacrebleu_eval.sh`.

## Citation
If you found this code or paper helpful, please consider citing:
```
@inproceedings{Weller2022PretrainedMF,
  title={Pretrained Models for Multilingual Federated Learning},
  author={Orion Weller and Marc Marone and Vladimir Braverman and Dawn J Lawrie and Benjamin Van Durme},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year={2022}
}
```

