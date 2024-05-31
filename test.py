import cv2
import vss_vision

frame = cv2.imread(filename="image.png")

res = vss_vision.run_seg(frame)
# print(res)

cv2.imwrite("res.png", res)
