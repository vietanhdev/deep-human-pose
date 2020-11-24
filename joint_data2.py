import json
from os.path import join
import random
import os

def write_label(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

def join_files(input_files, output_file, t=False):
    print(output_file)
    joint_set = []
    for ifile in input_files:
        with open(ifile, "r") as fp:
            data = json.load(fp)
            joint_set += data["labels"]
    pushing_items = [i for i in joint_set if i["is_pushing_up"]]
    not_pushing_items = [i for i in joint_set if 0 == i["is_pushing_up"]]
    if t:
        random.shuffle(pushing_items)
        random.shuffle(not_pushing_items)
        pushing_items = pushing_items[:1000]
        not_pushing_items = not_pushing_items[:1000]
        joint_set = not_pushing_items + pushing_items
    print("Pushing:", len(pushing_items))
    print("Not Pushing:", len(not_pushing_items))
    write_label({"labels": joint_set}, output_file)

join_files([
    # "data/3heads/train.json",
    "data/new_data/train.json",
], "data/combined/train.json")

join_files([
    # "data/3heads/val.json",
    "data/new_data/val.json",
], "data/combined/val.json", t=True)

join_files([
    # "data/3heads/test.json",
    "data/new_data/test.json",
], "data/combined/test.json", t=True)

# os.system("mkdir -p data/3heads/images")
# ofor i in data/new_data/images/*; do cp "$i"  data/3heads/images/; dones.system("cp -r data/3heads/images/* data/new_data/images/")
# os.system("cp -r data/new_data/images/* data/3heads/images/")
# # os.system("cp -r data/no_person/images/* data/3heads/images/")

# cp -r data/new_data/images/* data/3heads/images/

# cp -r data/new_data/images/* data/3heads/images/

# cp -r data/new_data/images/* data/3heads/images/

# cp -r data/new_data/images/* data/3heads/images/

