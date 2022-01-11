import cv2
import numpy as np
from tqdm import tqdm
import json

from swapface import swapface



if __name__ == '__main__' :
    # source video
    cap1 = cv2.VideoCapture("src/IMG_9582.mov")
    # destination video
    cap2 = cv2.VideoCapture("src/z.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('result/SBR-PE.avi', fourcc, 30.0, (1280,  720))

    pbar = tqdm(total = 300+1)
    Rects = []
    while cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 & ret2):
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame, r = swapface(frame1 , frame2, SBR=True, hm=False, pe=True)
        Rects.append(r)

        out.write(frame)
        pbar.update(1)
        if cv2.waitKey(1) == ord('q'):
            break

    Rects = np.array(Rects)
    x, y, w, h = np.max(np.array(Rects)[:, 0]), np.max(np.array(Rects)[:, 1]), np.max(np.array(Rects)[:, 2]), np.max(np.array(Rects)[:, 3])

    rect_dict = {"x": str(x), "y": str(y), "w": str(w), "h": str(h)}
    with open('result/rect.json', 'w') as f:
        json.dump(rect_dict, f)

    pbar.close()
    cap1.release()
    cap2.release()
    out.release()