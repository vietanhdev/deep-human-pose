import json
import cv2
import numpy as np

OUTPUT_IMG_FOLDER = "data/finetune/images/{}_{}.png"

with open("labels.json", "r") as fp:
    data = json.load(fp)

def preprocess_img(im, desired_size=800):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im


data = data["result"]["labels"]
new_data = []
for ii, label in enumerate(data):
    print(ii)
    frame_data = {}
    video_id = label["video_id"]
    path = "/mnt/DATA/PUSHUP_PROJECT/processed/{}.mp4".format(video_id)
    video = cv2.VideoCapture(path)
    if video is None:
        print("Error reading video")
        exit(0)
    label = json.loads(label["content"])["label"]
    frame_id = int(label["frame_id"])
    points = label["points"]
    if len(points) != 7:
        continue

    # for i in range(frame_id):
    #     video.read()
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    res, draw = video.read()
    if not res:
        print("Error")
        exit(0)

    draw = preprocess_img(draw)
    cv2.imwrite(OUTPUT_IMG_FOLDER.format(video_id, frame_id), draw)
    frame_data["image"] = "{}_{}.png".format(video_id, frame_id)
    points = np.array(points, dtype=float)

    frame_data["is_visible"] = [1,1,1,1,1,1,1]
    frame_data["points"] = []
    for i, point in enumerate(points):
        point = [int(point[0]), int(point[1])]
        frame_data["points"].append(point)
        draw = cv2.circle(draw, tuple(point), 3, (0,255,0), -1)
        draw = cv2.putText(draw,  str(i), (point[0]+10, point[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    new_data.append(frame_data)
    # cv2.imshow("Draw", draw)
    # cv2.waitKey(0)

with open("data/finetune/labels.json", "w") as fp:
    json.dump({"labels": new_data}, fp)