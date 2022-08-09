import os
import time
import glob
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(1)
random.seed(1)

def gather_data():
    all_data = {}
    lang_sets = ["en-fr", "ar-es", "ru-zh"]
    for lang_set in lang_sets:
        first, second = lang_set.split("-")
        all_data[lang_set] = {}
        for cur_lang in [first, second]:
            data = []
            with open(f"{lang_set}/UNv1.0.{lang_set}.{cur_lang}", "r") as fin:
                for line in fin:
                    data.append(line.strip())
                all_data[lang_set][cur_lang] = data
                print(f"Reading in {lang_set} {cur_lang} with length {len(data)}")
    return all_data

def split_and_save_data(save_path: str, min_size: int, data_w_text: dict):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for lang_set, data_dict in data_w_text.items():
        first, second = lang_set.split("-")
        data = pd.DataFrame({first: data_dict[first], second: data_dict[second]})
        print(f"Saving final data for lang {lang_set}")
        kept_data = data.sample(n=min_size, replace=False)
        train, dev_and_test = train_test_split(kept_data, test_size=10000)
        dev, test = train_test_split(dev_and_test, test_size=0.5)

        final_save_path = os.path.join(save_path, lang_set + "_train.csv")
        train.to_csv(final_save_path, index=False)
        dev.to_csv(final_save_path.replace("train", "dev"), index=False)
        test.to_csv(final_save_path.replace("train", "test"), index=False)
        print(f"Train shape {train.shape} and dev shape {dev.shape} and test shape {test.shape}")

if __name__ == "__main__":
    data_w_text = gather_data()
    split_and_save_data("splits_data/", 20000, data_w_text)
    # size following https://aclanthology.org/2020.wmt-1.4.pdf