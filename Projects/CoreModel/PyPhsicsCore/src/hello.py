import billiardsim as bs
import pybullet as p
import time

sim = bs.billiard_sim(
    gui_phys_server=True,
    ball_radius=0.0302,
    table_lateral_friction=0.33,
    cushion_restitution=0.86,
    cushion_lateral_friciton=0.07,
    cushion_anisotropic_friction=0.002,
    ball_restitution=0.98,
    ball_lateral_friciton=0.33,
    ball_rolling_friction=0,
    ball_spinning_friction=0,
    ball_anisotrofic_friction=0.5)

sim.locate(bs.BALL_RED1, (0.2, 0.33))
sim.locate(bs.BALL_RED2, (-0.2, 0.2))
sim.locate(bs.BALL_WHITE, (-0.4, -0.2))
sim.locate(bs.BALL_ORANGE, (0.7, 0.2))

# sim.hit(bs.BALL_ORANGE, 0, 2, do_visible_step=True)

while(1):
    if not sim.is_connected():
        break
    keys = p.getKeyboardEvents()
    for [key, state] in keys.items():
        if(key == ord('q') and state == p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN):
            sim.hit(bs.BALL_ORANGE, 1.11, 22, do_visible_step=True, simlength=11.0)
    time.sleep(0.033)
