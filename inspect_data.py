import json
import cv2
import numpy as np

with open("data/mpii/data.json") as fp:
    data = json.load(fp)

labels = data["labels"]
for label in labels:
    draw = np.zeros((1000, 1000, 3), dtype=np.uint8)

    for point_id, point in label["joint_pos"].items():
        print((int(point[0]), int(point[1])))
        draw = cv2.circle(draw, (int(point[0]), int(point[1])), 10, (0,255,0), -1)
        draw = cv2.putText(draw,  str(point_id), (int(point[0])+10, int(point[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA) 

    cv2.imshow("Debug", draw)
    cv2.waitKey(0)