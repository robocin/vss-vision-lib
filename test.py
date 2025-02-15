import cv2
import vss_vision
import numpy as np

frame = cv2.imread(filename="image.png")

res = vss_vision.run_seg(
    frame,
    np.array([42, 12, 147, 43, 0, 128, 233, 146]),
    np.array([range(1, 8)])
)

cv2.imwrite("res.png", res)

res = vss_vision.run_detect(
    frame,
    np.array([42, 12, 147, 43, 0, 128, 233, 146]),
    np.array([range(1, 8)])
)
print(res)

