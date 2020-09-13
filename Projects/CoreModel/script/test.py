# 가중치 함수 테스트
from matplotlib import pyplot as plt
import numpy as np
from math import *

x = np.arange(0.0, 1000.0, 1)
y1 = np.power(1.01, -x)
y2 = np.power(1.02, -x)
y3 = np.power(1.03, -x)

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.legend(['1.01', '1.02', '1.03'])
plt.show()
