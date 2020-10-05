# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

hsv_row = np.zeros((180, 3), np.uint8)
hsv_row[:, 0] = np.concatenate([np.array(range(165, 180)), np.array(range(0, 165))], axis=None)
hsv_row[:, 1] = 255
hsv_row[:, 2] = 255

hsv = np.array([hsv_row, ] * 60)
# hsv = cv2.resize(hsv, None, fx=2.0, fy=2.0)

hsv_as_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


plt.imshow(hsv_as_rgb)

# %%
