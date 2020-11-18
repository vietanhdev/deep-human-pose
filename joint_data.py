import json
import random
import os

def write_label(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

def join_files(input_files, output_file):
    joint_set = []
    for ifile in input_files:
        with open(ifile, "r") as fp:
            data = json.load(fp)
            joint_set += data["labels"]
    write_label({"labels": joint_set}, output_file)

join_files([
    "data/finetune/train.json",
    "data/mpii/train.json",
], "data/3heads/train.json")

join_files([
    "data/finetune/val.json",
    "data/mpii/val.json",
], "data/3heads/val.json")

join_files([
    # "data/finetune/test.json",
    "data/mpii/test.json",
], "data/3heads/test.json")


os.system("mkdir -p data/3heads/images")
os.system("cp -r data/finetune/images/* data/3heads/images/")
os.system("cp -r data/mpii/processed_images/* data/3heads/images/")
