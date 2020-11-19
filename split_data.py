import json
import random


with open("data/mpii/processed_data.json", "r") as fp:
    data = json.load(fp)
def write_label(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

random.seed(42)
random.shuffle(data)
t1 = int(0.94*len(data))
t2 = int(0.97*len(data))

for d in data:
    d.pop("is_visible", None)
    d["is_pushing_up"] = False
    d["contains_person"] = True

train = data[:t1]
val = data[t1:t2]
test = data[t2:]

write_label({"labels": train}, "data/mpii/train.json")
write_label({"labels": val}, "data/mpii/val.json")
write_label({"labels": test}, "data/mpii/test.json")

print("train", len(train))
print("val", len(val))
print("test", len(test))