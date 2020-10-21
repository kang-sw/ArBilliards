import numpy as np

felt_width = 1735/2
felt_height = 915/2
width_offset_1 = 15
width_offset_2 = 15
height_offset_1 = 1
height_offset_2 = 1
marker_offset = 13
marker_step = 205

l = []

for i in range(2, -3, -1):
    l.append([-(felt_width + marker_offset) * 0.001, 0, (width_offset_1 + i * marker_step) * 0.001])

for i in range(-4, 5, 1):
    l.append([(height_offset_1 + i * marker_step) * 0.001, 0, -(felt_height + marker_offset) * 0.001])

for i in range(-2, 3, 1):
    l.append([(felt_width + marker_offset) * 0.001, 0, (width_offset_2+i * marker_step) * 0.001])

for i in range(4, -5, -1):
    l.append([(height_offset_2 + i * marker_step) * 0.001, 0, (felt_height + marker_offset) * 0.001])


print(l)
