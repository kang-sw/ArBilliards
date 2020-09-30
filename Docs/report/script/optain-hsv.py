import cv2
import numpy as np
from tkinter import filedialog

filename = filedialog.askopenfilename()

if len(filename) == 0:
    exit()

img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

channels = cv2.split(hsv)

channels[0] = channels[0] + 5
maskmat = np.uint8((channels[0] > 180)*180)


hsv = cv2.merge(channels)

filtered = cv2.inRange(hsv, np.uint8([0, 160, 0]), np.uint8([25, 255, 255]))
edge = filtered - cv2.erode(filtered, None)

ctrs = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


ctr = max(ctrs[0], key=cv2.contourArea)
ctr = cv2.approxPolyDP(ctr, 10, True)
ctr = cv2.convexHull(ctr, clockwise=False)
ctr = cv2.approxPolyDP(ctr, 10, True)
debug_src = cv2.drawContours(img, [ctr], -1, [0, 0, 0], thickness=2)
for vtx in ctr:
    pt = vtx[0]
    cv2.circle(debug_src, (pt[0], pt[1]), 10, [0, 0, 255], thickness=-1)


cv2.imshow("render", debug_src)
# cv2.imshow("h", channels[0] - maskmat)
# cv2.imshow("s", channels[1])
# cv2.imshow("v", channels[2])
cv2.imshow("filtered", filtered)
cv2.imshow("edge", edge)
cv2.waitKey(0)
