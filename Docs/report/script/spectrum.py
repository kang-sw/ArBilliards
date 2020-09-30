# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

hsv_row = np.zeros((180, 3), np.uint8)
hsv_row[:, 0] = np.concatenate([np.array(range(175, 180)), np.array(range(0, 175))], axis=None)
hsv_row[:, 1] = 255
hsv_row[:, 2] = 255

hsv = np.array([hsv_row, ] * 60)

hsv_as_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


plt.imshow(hsv_as_rgb)

# %%
