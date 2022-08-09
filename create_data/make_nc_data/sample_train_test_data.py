import os
import time
import glob
import json
import random
import pandas as pd
import numpy as np

np.random.seed(1)
random.seed(1)

labels = {
    "finance": 0,
    "entertainment": 1,
    "sports": 2,
    "news": 3,
    "autos": 4,
    "video": 5,
    "lifestyle": 6,
    "travel": 7,
    "health": 8,
    "foodanddrink": 9,
}

def read_tsv(file_path: str, skip_first=False):
    data = []
    with open(file_path, "r") as fin:
        for idx, line in enumerate(fin):
            if skip_first and not idx:
                continue
            skip_flag = False
            segments = line.strip().split("\t")
            assert len(segments) == 3, segments
            if not skip_flag:
                data.append({
                    "input": segments[0] + segments[1], # combine news title with news body
                    "label": labels[segments[2]],
                })
    return pd.DataFrame(data)


# query \t news title \t news body \t news category
def gather_data():
    num_for_train = 8000
    num_for_dev_test = 1000
    save_path = "splits_data"
    datasets = []
    lang_ids = ["de", "en", "es", "fr", "ru"]
    for lang_id in lang_ids:
        dev = read_tsv(f"xglue_full_dataset/NC/xglue.nc.{lang_id}.dev") # test doesn't have labels / no train for other langs
        datasets.append((lang_id, dev))

    # downsample training sets to simulate FL scenario
    for (lang_id, dev) in datasets:
        print(lang_id, "saving to file")
        save_path = f"nc/{lang_id}"
        if not os.path.isdir("nc"):
            os.makedirs("nc")

        all_data = dev.sample(frac=1)
        train_sampled = all_data.iloc[:num_for_train]
        dev = all_data.iloc[num_for_train:num_for_train+num_for_dev_test]
        test = all_data.iloc[num_for_train+num_for_dev_test : ]
        dev.to_csv(save_path + "_dev.csv", index=None)
        test.to_csv(save_path + "_test.csv", index=None)
        train_sampled.to_csv(save_path + "_train.csv", index=None)

        print(f"train_sampled shape {train_sampled.shape}")
        print(f"dev shape {dev.shape}")
        print(f"test shape {test.shape}")

    return {}


if __name__ == "__main__":
    data_w_text = gather_data()
