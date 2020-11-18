import json
import cv2

IMAGE_PATH_PATTERN = ""

with open("labels.json", "r") as fp:
    data = json.load(fp)


data = data["result"]["labels"]
for label in data:
    video_id = label["video_id"]
    label = json.loads(label["content"])["label"]
    frame_id = label["frame_id"]
    points = label["points"]
    draw = cv2.imread(IMAGE_PATH_PATTERN.format(video_id, frame_id))
    for i, point in enumerate(points):
        draw = cv2.circle(draw, tuple(point), 3, (0,255,0), -1)
        draw = cv2.putText(draw,  str(i), (point[0]+10, point[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Draw", draw)
    cv2.waitKey(0)
    