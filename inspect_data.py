import json
import cv2
import numpy as np
import os


output_img_folder = "data/mpii/processed_images"

with open("data/mpii/data.json") as fp:
    data = json.load(fp)
labels = data["labels"]

# Remove images with more than 1 person
name_freq = {}
for label in labels:
    filename = label["filename"]
    if filename not in name_freq.keys():
        name_freq[filename] = 1
    else:
        name_freq[filename] += 1
labels = [label for label in labels if name_freq[label["filename"]] == 1]


print(len(labels))

new_labels = []
for i, label in enumerate(labels):
    # draw = np.zeros((1000, 1000, 3), dtype=np.uint8)
    draw = cv2.imread(os.path.join("data/mpii/images/", label["filename"]))
    min_x = int(min(p[0] for p in label["joint_pos"].values()))
    min_y = int(min(p[1] for p in label["joint_pos"].values()))
    max_x = int(max(p[0] for p in label["joint_pos"].values()))
    max_y = int(max(p[1] for p in label["joint_pos"].values()))
    pad = 100
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(draw.shape[1]-1, max_x + pad)
    max_y = min(draw.shape[1]-1, max_y + pad)
    draw = draw[min_y:max_y, min_x:max_x]
    # cv2.imwrite(os.path.join(output_img_folder, str(i)+".png"), draw)
    for p in label["joint_pos"].keys():
        label["joint_pos"][p][0] -= min_x
        label["joint_pos"][p][1] -= min_y
    for point_id, point in label["joint_pos"].items():
        # print((int(point[0]), int(point[1])))
        color = (0, 255, 0) if label["is_visible"][point_id] == 1 else (0, 0, 255)
        draw = cv2.circle(draw, (int(point[0]), int(point[1])), 10, color, -1)
        draw = cv2.putText(draw,  str(point_id), (int(point[0])+10, int(point[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA) 

    new_label = {}
    new_label["image"] = str(i)+".png"
    new_label["points"] = []
    new_label["is_visible"] = []
    for p in ["10", "11", "12", "9", "13", "14", "15"]:
        point = label["joint_pos"][p]
        new_label["points"].append([int(point[0]), int(point[1])])
        new_label["is_visible"].append(label["is_visible"][p])
    new_labels.append(new_label)

    print(i)
    cv2.imshow("Debug", draw)
    cv2.waitKey(0)

# def write_label(data, file_path):
#     with open(file_path, 'w') as outfile:
#         json.dump(data, outfile)
# write_label(new_labels, "data/mpii/processed_data.json")