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

def gather_all_files(folder_path: str) -> dict:
    all_files = {}
    for file in glob.glob(os.path.join(folder_path, "europarl-v9.*")):
        if "gz" in file:
            continue
        print(f"Loading {file}")
        lang = file.split("/")[-1].split(".")[-1]
        data = []
        with open(file, "r") as fin:
            for line in fin:
                data.append(line)
        all_files[lang] = data
    
    return all_files

def preprocess_str(x: str) -> str:
    x = x.strip()
    if len(x) < 3 or len(x.split(" ")) < 3:
       return ""
    return x

def preprocess_data(data_w_text: dict) -> dict:
    new_data = {}
    for lang, data in data_w_text.items():
        data_non_nan = [line for line in data if line not in ["", None]]
        data_non_nan = [preprocess_str(x) for x in data_non_nan]
        data_non_nan = [line for line in data_non_nan if line not in [""]]
        new_data[lang] = pd.DataFrame({"text": data_non_nan})
    return new_data

def split_and_save_data(save_path: str, min_size: int, data_w_text: dict, test_percent: float):
    for lang, data in data_w_text.items():
        print(f"Saving final data for lang {lang}")
        kept_data = data.sample(n=min_size, replace=False)
        train, dev_and_test = train_test_split(kept_data, test_size=test_percent)
        dev, test = train_test_split(dev_and_test, test_size=0.5)

        final_save_path = os.path.join(save_path, lang + "_train.csv")
        train.to_csv(final_save_path, index=False)
        dev.to_csv(final_save_path.replace("train", "dev"), index=False)
        test.to_csv(final_save_path.replace("train", "test"), index=False)
        print(f"Train shape {train.shape} and dev shape {dev.shape} and test shape {test.shape}")
    
def randomly_select_and_split(file_path: str, test_percent: float = 0.333333):
    save_path = "splits_data"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    data_w_text = gather_all_files(file_path)
    data_w_text = preprocess_data(data_w_text)
    info = [(lang, len(data)) for lang, data in data_w_text.items() if data is not None and not data.empty]
    min_size = min([30000] + [len(data) for _, data in data_w_text.items() if data is not None and not data.empty])
    print(f"Minimum Data Size is {min_size}")
    split_and_save_data(os.path.join(file_path, save_path), min_size, data_w_text, test_percent)


if __name__ == "__main__":
    randomly_select_and_split("./")
