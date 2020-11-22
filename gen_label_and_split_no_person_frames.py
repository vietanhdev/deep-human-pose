import os
import json
import random

# os.system("mkdir -p data/no_person/images")
image_folder = "data/no_person/images"

data = []
images = os.listdir(image_folder)
for image in images:
    if not image.endswith("png"):
        continue
    label = {}
    label["points"] = [[-1, -1],[-1, -1],[-1, -1],[-1, -1],[-1, -1],[-1, -1],[-1, -1]]
    label["is_pushing_up"] = False
    label["contains_person"] = False
    label["image"] = image
    data.append(label)

def write_label(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

random.seed(42)
random.shuffle(data)
t1 = int(0.8*len(data))
t2 = int(0.9*len(data))

train = data[:t1]
val = data[t1:t2]
test = data[t2:]

write_label({"labels": train}, "data/no_person/train.json")
write_label({"labels": val}, "data/no_person/val.json")
write_label({"labels": test}, "data/no_person/test.json")

print("train", len(train))
print("val", len(val))
print("test", len(test))