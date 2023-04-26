from tokenize import Double
import numpy as np
import utils.bezier_interpolation as bezier
import utils.foot_trajectory_generator as ftg
import matplotlib.pyplot as plt

alpha = np.array([0.5, 0.0, -0.3, 0., 0.2, -0.])
t_step = 1000

q = []
dq = []

# for i in range(t_step):
#     q.append(bezier.bezier_get_joint_poision((i/t_step),alpha))
#     dq.append(bezier.bezier_get_joint_vel((i/t_step),alpha))

_ftg = ftg.FootTrajectoryGenerator(T=0.4,max_foot_height=0.1)
height_data = []
for i in range(1000):
    fi = -0.3

    h = _ftg.setPhasesAndGetDeltaHeights(i/1000, fi)
    height_data.append(h)
fig, ax= plt.subplots()
t = np.linspace(0, 1,1000)

ax.plot(t, height_data, label='q')
# ax.plot(t, dq, label='dq')
ax.legend()
plt.show()