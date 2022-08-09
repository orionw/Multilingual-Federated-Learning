import glob
import os
import json
import tqdm
from pathlib import Path
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
import xml
import numpy as np
import random
random.seed(3)

LANGS = [ "en", "ar", "es", "fr", "ru", "zh"]

def read_in_text(xml_file: str):
    lines = []
    with open(xml_file, "r") as fin:
        try:
            root = ET.parse(fin).getroot()
            for p_tag in root.findall('text')[0].find("body").findall("p"):
                for s_tag in p_tag.findall("s"):
                    if s_tag.text is not None:
                        lines.append(s_tag.text)
        except xml.etree.ElementTree.ParseError:
            print(f"Cannot parse {xml_file}")
    if "zh" in xml_file[:10]:
        return [line for line in lines if len(line) > 5] 
    else:
        return [line for line in lines if len(line.split(' ')) > 5]

def get_paths(lang_path: str) -> list:
    all_paths = []
    for file_path in tqdm.tqdm(Path(lang_path).rglob(os.path.join("*.xml"))):
        relative_path = "/".join(str(file_path).split("/")[2:])
        all_paths.append(relative_path)
    assert len(all_paths) == len(set(all_paths)), f"{len(all_paths)} {len(set(all_paths))}"
    return all_paths

def gather_exclusive_paths(base_path: str):
    lang_map = {}
    total_docs = 0
    for lang in LANGS:
        print(f'Gathering paths for lang {lang}')
        lang_map[lang] = get_paths(os.path.join(base_path, lang))
        total_docs += len(lang_map[lang])
    
    print(f"The number of total docs is {total_docs}")
    assert total_docs == 799276, f"Expected 799,276 documents - got {total_docs}"
    
    reverse_dict = defaultdict(list)
    for lang, paths in lang_map.items():
        for file_path in paths:
            reverse_dict[file_path].append(lang)

    langs_num = [len(lang_list) for _, lang_list in reverse_dict.items()]
    lang_counter = Counter(langs_num)
    print(f"Average langs={np.mean(langs_num)} for {lang_counter} with total unique={len(reverse_dict)}")

    exclusive_map = defaultdict(list)
    for file_path, langs_available in reverse_dict.items():
        # if there are multiple options for languages for a document, randomly choose
        if len(langs_available) == 6 and False:
            lang_to_use = random.choices(
                population=["ar", "en", "es", "fr", "ru", "zh"],
                weights=[0.23669934, 0.06437802, 0.18635274, 0.06519098, 0.16047007, 0.28690884],
                k=1
            )
            # np.array([20909,16215,37016,36719,27827,25134])
            # ar en es fr ru zh
# array([0.12763399, 0.09898059, 0.22595532, 0.22414235, 0.16986326,
#        0.15342449])
#             [20909,16215,37016,36719,27827,25134] / 6
#   (1/6) + ((1/6) - (arr / 163820))
        else:
            lang_to_use = random.sample(langs_available, 1)
        assert len(lang_to_use) == 1
        # derive the full path back
        exclusive_map[lang_to_use[0]].append(f"UNv1.0-TEI/{lang_to_use[0]}/{file_path}")
    
    # sanity check here that they are unique
    total = []
    for lang, paths in exclusive_map.items():
        lang_data = list(exclusive_map[lang])
        exclusive_map[lang] = lang_data
        print(f"Lang {lang} has {len(lang_data)} unique items")
        total.extend(lang_data)
    assert len(total) == len(set(total)), f"{len(total)} vs {len(set(total))}"

    with open("exclusive_dict.json", "w") as fout:
        json.dump(exclusive_map, fout, indent=4)

    for lang, paths in exclusive_map.items():
        print(f"Gathering lines for {lang}")
        data = []
        for file_path in tqdm.tqdm(paths):
            data.extend(read_in_text(file_path))
        print(f"Has {len(data)} lines")
        with open(f"{lang}.txt", "w") as fout:
            for line in data:
                fout.write(line + "\n")


if __name__ == "__main__":
    gather_exclusive_paths("UNv1.0-TEI")