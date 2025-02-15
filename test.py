import cv2
import vss_vision
import numpy as np

frame = cv2.imread(filename="image_2.png")

cv2.resize(frame, (640, 480))

res = vss_vision.run_seg(
    frame,
    np.array([81, 27, 151, 59, 6, 94, 248, 120]),
    np.array(range(1, 8))
)

cv2.imwrite("res.png", res)

res = vss_vision.run_detect(
    frame,
    np.array([81, 27, 151, 59, 6, 94, 248, 120]),
    np.array(range(1, 8)),
    np.array([153, 1000])
)

image = res["image"]
gameInfo = res["gameInfo"]

cv2.imwrite("res_detect.png", image)
print(gameInfo)


