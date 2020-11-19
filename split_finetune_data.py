import json
import random


with open("data/finetune/labels.json", "r") as fp:
    data = json.load(fp)["labels"]
def write_label(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

# Dont shuffle finetune data to prevent split images from same video
# random.seed(42)
# random.shuffle(data)

t1 = int(0.7*len(data))
t2 = int(0.85*len(data))

for d in data:
    d.pop("is_visible", None)
    d["is_pushing_up"] = True
    d["contains_person"] = True

train = data[:t1] + data[:t1]
val = data[t1:t2]
test = data[t2:]

write_label({"labels": train}, "data/finetune/train.json")
write_label({"labels": val}, "data/finetune/val.json")
write_label({"labels": test}, "data/finetune/test.json")

print("train", len(train))
print("val", len(val))
print("test", len(test))