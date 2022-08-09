import os
import time
import glob
import json
import random
import pandas as pd
import numpy as np

np.random.seed(1)
random.seed(1)

def read_tsv(file_path: str, trg_lang: str):
    data = []
    with open(file_path, "r") as fin:
        for line in fin:
            segments = line.strip().split("\t")
            if len(segments) != 3:
                if len(segments) == 2 and len(segments[0].split(" ")) > 1:
                    data.append({
                        "id": -1,
                        "en": segments[0],
                        trg_lang: segments[1]
                    })
                else:
                    print(file_path, "error")
                    breakpoint()
                    raise Exception(segments)
            else:
                data.append({
                    "id": segments[0],
                    "en": segments[1],
                    trg_lang: segments[2]
                })
    return pd.DataFrame(data)

def gather_data():
    save_path = "splits_data"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    en_ja = read_tsv("MTNT/train/train.en-ja.tsv.corrected", trg_lang="ja")
    en_fr = read_tsv("MTNT/train/train.en-fr.tsv.corrected", trg_lang="fr")

    # remove super short sentences that are numbers or emojis
    en_ja_short = en_ja.apply(lambda x: len(x['en'].split(" ")) < 3, axis=1)
    en_ja = en_ja[~en_ja_short]

    en_ja_only = set(en_ja["en"].to_list())
    en_fr_matched_bool = en_fr.apply(lambda x: x["en"] in en_ja_only, axis=1)
    matched_en_fr = en_fr[en_fr_matched_bool]
    not_matched_en_fr = en_fr[~en_fr_matched_bool]

    num_to_random_sample = len(en_ja) - len(matched_en_fr)
    additional_samples = not_matched_en_fr.sample(n=num_to_random_sample)
    full_en_fr = pd.concat([matched_en_fr, additional_samples])

    full_en_fr = full_en_fr[full_en_fr.columns[1:]]
    en_ja = en_ja[en_ja.columns[1:]]

    en_ja.to_csv(os.path.join(save_path, "en-ja_train.csv"), index=None)
    full_en_fr.to_csv(os.path.join(save_path, "en-fr_train.csv"), index=None)
    print(f"En-Ja Train shape {en_ja.shape} ")
    print(f"En-Fr Train shape {full_en_fr.shape} ")


    testing_files = [
        ("MTNT/valid/valid.en-fr.tsv.corrected", "en-fr_dev.csv"),
        ("MTNT/valid/valid.en-ja.tsv.corrected", "en-ja_dev.csv"),
        ("MTNT/test/test.en-fr.tsv.corrected", "en-fr_test.csv"),
        ("MTNT/test/test.en-ja.tsv.corrected", "en-ja_test.csv"),
    ]
    for file_path, save_name in testing_files:
        trg_lang = save_name.split("-")[1].split("_")[0]
        data = read_tsv(file_path, trg_lang)
        data = data[data.columns[1:]]
        data.to_csv(os.path.join(save_path, save_name), index=None)
        print(f"{save_name} shape {data.shape}")

    return {}


if __name__ == "__main__":
    data_w_text = gather_data()
    # size following https://aclanthology.org/2020.wmt-1.4.pdf