# Simulator class
import pybullet as p
import math as m
import numpy as np
import time

BALL_RED1 = 0
BALL_RED2 = 1
BALL_ORANGE = 2
BALL_WHITE = 3


class billiard_sim:
    __refcnt = 0

    def __init__(self, rule_4ball=True,
                 ball_radius=1.0,
                 table_size=(1.735, 0.85),
                 gui_phys_server=False,
                 ball_mass=0.210,
                 sim_steps=0.01,
                 table_lateral_friction=0.1,
                 cushion_lateral_friciton=0.1,
                 cushion_restitution=0.87,
                 cushion_anisotropic_friction=0,
                 ball_lateral_friciton=0.1,
                 ball_restitution=0.91,
                 ball_rolling_friction=0.02,
                 ball_spinning_friction=0.02,
                 ball_anisotrofic_friction=0):
        billiard_sim.__refcnt += 1

        if(p.isConnected() == False):
            p.connect(p.GUI if gui_phys_server else p.DIRECT)
            p.setGravity(0, 0, -9.8)
            p.setTimeStep(sim_steps)
            self.__sim_step = sim_steps
            self.__is_gui_mode = gui_phys_server
        else:
            raise Exception("Two simulation instance cannot exist at the same time.")

        # initialize land shape
        landShape = p.createCollisionShape(p.GEOM_PLANE)
        landId = p.createMultiBody(0, landShape, -1)
        print(p.getDynamicsInfo(landId, -1))
        p.changeDynamics(landId, -1, lateralFriction=table_lateral_friction)

        # initialize table cushions
        longer = max(table_size)
        cushionShape = p.createCollisionShape(p.GEOM_BOX,
                                              halfExtents=[longer/2, longer/2, ball_radius],
                                              flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        hlong = longer/2
        w = table_size[0] / 2
        h = table_size[1] / 2
        # 네 개 쿠션의 중심 좌표입니다. z축에 대해 정사각형인 큐브를 배치하므로, longer만큼의 거리를 추가로 오프셋
        # 좌, 우, 상, 하 순서
        origins = [[-w-hlong, 0], [w+hlong, 0], [0, -h-hlong], [0, h+hlong]]
        for coord in origins:
            cushionId = p.createMultiBody(0, cushionShape, -1, [coord[0], coord[1], ball_radius])
            print(p.getDynamicsInfo(cushionId, -1))
            p.changeDynamics(cushionId, -1,
                             lateralFriction=cushion_lateral_friciton,
                             restitution=cushion_restitution,
                             # anisotropicFriction=cushion_anisotropic_friction
                             )

        # initialize balls. 3-ball currently not supported.
        if not rule_4ball:
            raise Exception("Currently 3-ball rule is not supported!")

        self.__ballId = [None]*4
        self.__ball_radius = ball_radius
        ballShape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        colors = [[1, 0, 0, 1], [1, 0, 0, 1], [0.88, 0.66, 0, 1], [1, 1, 1, 1]]
        for bidx in range(4):
            print("spawn ball")
            ballVisShape = p.createVisualShape(p.GEOM_SPHERE, ball_radius, rgbaColor=colors[bidx])
            ballId = p.createMultiBody(ball_mass, ballShape, -1, [0, 0, ball_radius])
            print(p.getDynamicsInfo(ballId, -1))
            p.changeDynamics(ballId, -1,
                             # frictionAnchor=0,
                             lateralFriction=ball_lateral_friciton,
                             restitution=ball_restitution,
                             rollingFriction=ball_rolling_friction,
                             spinningFriction=ball_spinning_friction,
                             # anisotropicFriction=ball_anisotrofic_friction,
                             )

            self.__ballId[bidx] = ballId
            pass

        self.__ball_init_pos = [np.array([0, 0])]*4

    def __del__(self):
        billiard_sim.__refcnt -= 1

    def locate(self, ballindex, pos):
        self.__ball_init_pos[ballindex] = np.array(pos)
        pass

    def hit(self, ballindex, dir_rad, force, ofst_h_rad=0, simlength=10.0, do_visible_step=False):
        # 모든 공을 원점 위치로 되돌립니다.
        for bidx in range(4):
            pos = self.__ball_init_pos[bidx]
            p.resetBasePositionAndOrientation(self.__ballId[bidx], [pos[0], pos[1], self.__ball_radius], [0, 0, 0, 1])

        # x축이 힘을 가하는 기준 방향입니다.
        point = np.array([-1, 0]).T  # 당점으로, 기본적으로 힘의 반대 방향을 가리킵니다.
        direction = np.array([1, 0]).T  # 공이 나아갈 방향입니다.

        # 좌우 당점을 주기 위해 오프셋(공의 표면을 따라 point를 회전)

        # 회전 행렬 계산
        c, s = m.cos(dir_rad), m.sin(dir_rad)
        R = np.array(((c, -s), (s, c)))
        dir_f = (R @ direction) * force

        p.applyExternalForce(self.__ballId[ballindex], -1, [*dir_f, 0], [0, 0, 0], flags=p.WORLD_FRAME)

        # 주어진 시간동안 시뮬레이션 수행
        for T in np.arange(0, simlength, self.__sim_step):
            p.stepSimulation()
            tget_time = time.perf_counter() + self.__sim_step
            while(time.perf_counter() < tget_time):
                time.sleep(0)
                pass

    def retrieve_hit_result(self):
        pass

    def is_connected(self):
        return p.isConnected()
