import math as m
import numpy as np

dir_rad = 3.141592
direction = np.array([[1, 0]]).T
c, s = m.cos(dir_rad), m.sin(dir_rad)
R = np.array(((c, -s), (s, c)))

v = R @ direction

print(R)
print(direction)
print(v)
