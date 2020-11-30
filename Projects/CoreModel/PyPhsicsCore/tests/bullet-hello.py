import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.createCollisionShape(p.GEOM_PLANE)

sphereShape = p.createCollisionShape(p.GEOM_SPHERE, radius=1)
colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])

p.createMultiBody(1, sphereShape, -1, [0, 0, 10], [0, 0, 0, 1])
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)
while (1):
    if not p.isConnected():
        break

    time.sleep(0.1)
