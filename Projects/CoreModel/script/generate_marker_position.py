import numpy as np

felt_width = 1735/2
felt_height = 915/2
marker_offset = 12
marker_step = 205

l = []

for i in range(2, -3, -1):
    l.append([-(felt_width + marker_offset), 0, i * marker_step])

for i in range(-4, 5, 1):
    l.append([i * marker_step, 0, -(felt_height + marker_offset)])

for i in range(-2, 3, 1):
    l.append([(felt_width + marker_offset), 0, i * marker_step])

for i in range(4, -5, -1):
    l.append([i * marker_step, 0, (felt_height + marker_offset)])


print('{', end='')
for v in l:
    print(f'{{ {0.001*v[0]:.4f}, {0.001*v[1]:.4f}, {0.001*v[2]:.4f} }}, ', end='')
print('}')
