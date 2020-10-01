# %%
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

channels[0] = channels[0] + 15
maskmat = np.uint8((channels[0] > 180)*180)
channels[0] = channels[0] - maskmat

hsv = cv2.merge(channels)

filtered = cv2.inRange(hsv, np.int32([0, 150, 0]), np.int32([30, 255, 255]))
filtered[0, :] = 0
filtered[filtered.shape[0]-1, :] = 0
filtered[:, 0] = 0
filtered[:, filtered.shape[1]-1] = 0

edge = filtered - cv2.erode(filtered, None)

ctrs = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ctr = max(ctrs[0], key=cv2.contourArea)
ctr = cv2.approxPolyDP(ctr, 5, True)
ctr = cv2.convexHull(ctr, clockwise=False)
ctr = cv2.approxPolyDP(ctr, 5, True)
debug_src = cv2.drawContours(img, [ctr], -1, [0, 0, 0], thickness=2)
for vtx in ctr:
    pt = vtx[0]
    cv2.circle(debug_src, (pt[0], pt[1]), 10, [0, 0, 255], thickness=-1)

canvas = np.ndarray(img.shape, dtype=np.uint8)
canvas.fill(255)
cv2.drawContours(canvas, [ctr], -1, (0, 0, 0), 15)
cv2.drawContours(canvas, [ctr], -1, (233, 11, 32), -1)
cv2.rectangle(canvas, (0, 0), canvas.shape[::-1][1:3], (0, 0, 0), 10)

# %%
cv2.imshow("canvas", canvas)
cv2.imshow("render", debug_src)
# cv2.imshow("h", channels[0] - maskmat)
# cv2.imshow("s", channels[1])
# cv2.imshow("v", channels[2])
cv2.imshow("filtered", filtered)
# cv2.imshow("edge", edge)
cv2.waitKey(0)
