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
    all_files = {}
    for file_path in glob.glob("*.txt"):
        lang = file_path.split("/")[-1].split(".")[0]
        data = []
        with open(file_path, "r") as fin:
            for line in fin:
                data.append(line.strip())
        all_files[lang] = data
        print(f"Reading in {file_path} with length {len(data)}")
    return all_files

def split_and_save_data(save_path: str, min_size: int, data_w_text: dict):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for lang, data in data_w_text.items():
        data = pd.DataFrame({"text": data})
        print(f"Saving final data for lang {lang}")
        kept_data = data.sample(n=min_size, replace=False)
        train, dev_and_test = train_test_split(kept_data, test_size=10000)
        dev, test = train_test_split(dev_and_test, test_size=0.5)

        final_save_path = os.path.join(save_path, lang + "_train.csv")
        train.to_csv(final_save_path, index=False)
        dev.to_csv(final_save_path.replace("train", "dev"), index=False)
        test.to_csv(final_save_path.replace("train", "test"), index=False)
        print(f"Train shape {train.shape} and dev shape {dev.shape} and test shape {test.shape}")

if __name__ == "__main__":
    data_w_text = gather_data()
    split_and_save_data("splits_data/", 60000, data_w_text)